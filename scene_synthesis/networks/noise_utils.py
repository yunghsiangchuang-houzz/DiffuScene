"""
Noise utilities for deterministic diffusion sampling across multiple checkpoints.

Usage:
    # Pre-generate noise cache for a scene
    noise_cache = DiffusionNoiseCache.generate(
        seed=42,
        batch_size=6,
        num_points=50,
        point_dim=17,
        num_timesteps=1000,
        partial_boxes_shape=(6, 40, 17),
        device='cpu'
    )
    
    # Save for later use
    noise_cache.save('noise_cache_scene123.pt')
    
    # Load and use
    noise_cache = DiffusionNoiseCache.load('noise_cache_scene123.pt')
    noise_cache = noise_cache.to('cuda')
    
    # Pass to network
    result = network.complete_scene(..., noise_cache=noise_cache)
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DiffusionNoiseCache:
    """
    Pre-generated noise tensors for deterministic diffusion sampling.
    
    Stores:
    - initial_noise: Starting noise for diffusion (batch_size, num_points, point_dim)
    - timestep_noises: Dict[int, Tensor] - noise for each timestep during denoising
    - partial_noises: Dict[int, Tensor] - noise for partial boxes re-noising at each timestep
    """
    initial_noise: torch.Tensor
    timestep_noises: dict  # {t: noise_tensor}
    partial_noises: dict   # {t: noise_tensor} 
    seed: int
    device: torch.device
    
    @classmethod
    def generate(
        cls,
        seed: int,
        batch_size: int,
        num_points: int,
        point_dim: int,
        num_timesteps: int,
        partial_boxes_shape: Optional[Tuple[int, int, int]] = None,
        device: str = 'cpu'
    ) -> 'DiffusionNoiseCache':
        """
        Generate all noise tensors needed for one complete diffusion sampling.
        
        Args:
            seed: Random seed for reproducibility
            batch_size: Number of samples in batch (e.g., n_samples per scene)
            num_points: Number of points per sample (e.g., sample_num_points=50)
            point_dim: Dimension of each point (e.g., 17 for bbox + class + objectness)
            num_timesteps: Number of diffusion timesteps (e.g., 1000)
            partial_boxes_shape: Shape of partial boxes (batch_size, partial_num_points, point_dim)
            device: Device to generate noise on ('cpu' recommended for reproducibility)
        
        Returns:
            DiffusionNoiseCache with all pre-generated noise tensors
        """
        # Use CPU generator for reproducibility (GPU generators can vary across devices)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        
        sample_shape = (batch_size, num_points, point_dim)
        
        # 1. Initial noise
        initial_noise = torch.randn(sample_shape, generator=gen)
        
        # 2. Per-timestep noise for denoising loop (reverse order: T-1 down to 0)
        timestep_noises = {}
        for t in reversed(range(num_timesteps)):
            timestep_noises[t] = torch.randn(sample_shape, generator=gen)
        
        # 3. Per-timestep noise for partial boxes re-noising
        partial_noises = {}
        if partial_boxes_shape is not None:
            for t in reversed(range(num_timesteps)):
                partial_noises[t] = torch.randn(partial_boxes_shape, generator=gen)
        
        return cls(
            initial_noise=initial_noise,
            timestep_noises=timestep_noises,
            partial_noises=partial_noises,
            seed=seed,
            device=torch.device(device)
        )
    
    def to(self, device) -> 'DiffusionNoiseCache':
        """Move all noise tensors to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        return DiffusionNoiseCache(
            initial_noise=self.initial_noise.to(device),
            timestep_noises={t: n.to(device) for t, n in self.timestep_noises.items()},
            partial_noises={t: n.to(device) for t, n in self.partial_noises.items()},
            seed=self.seed,
            device=device
        )
    
    def save(self, path: str):
        """Save noise cache to file for reproducibility."""
        torch.save({
            'initial_noise': self.initial_noise.cpu(),
            'timestep_noises': {t: n.cpu() for t, n in self.timestep_noises.items()},
            'partial_noises': {t: n.cpu() for t, n in self.partial_noises.items()},
            'seed': self.seed,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DiffusionNoiseCache':
        """Load noise cache from file."""
        data = torch.load(path, map_location='cpu')
        cache = cls(
            initial_noise=data['initial_noise'],
            timestep_noises=data['timestep_noises'],
            partial_noises=data['partial_noises'],
            seed=data['seed'],
            device=torch.device('cpu')
        )
        if device != 'cpu':
            cache = cache.to(device)
        return cache
    
    def clone(self) -> 'DiffusionNoiseCache':
        """Create a deep copy of the noise cache."""
        return DiffusionNoiseCache(
            initial_noise=self.initial_noise.clone(),
            timestep_noises={t: n.clone() for t, n in self.timestep_noises.items()},
            partial_noises={t: n.clone() for t, n in self.partial_noises.items()},
            seed=self.seed,
            device=self.device
        )
    
    @property
    def batch_size(self) -> int:
        """Return the batch size this cache was generated for."""
        return self.initial_noise.shape[0]
    
    def slice(self, n_samples: int) -> 'DiffusionNoiseCache':
        """
        Return a new cache with only the first n_samples.
        
        Use this when you want to use fewer samples than the cache was generated for.
        This maintains determinism - sample 0 will always get the same noise.
        
        Args:
            n_samples: Number of samples to keep (must be <= self.batch_size)
        
        Returns:
            New DiffusionNoiseCache with sliced tensors
        
        Raises:
            ValueError if n_samples > self.batch_size
        """
        if n_samples > self.batch_size:
            raise ValueError(
                f"Cannot slice {n_samples} samples from cache with batch_size={self.batch_size}. "
                f"Regenerate cache with --n_samples>={n_samples}"
            )
        
        if n_samples == self.batch_size:
            return self.clone()
        
        return DiffusionNoiseCache(
            initial_noise=self.initial_noise[:n_samples].clone(),
            timestep_noises={t: n[:n_samples].clone() for t, n in self.timestep_noises.items()},
            partial_noises={t: n[:n_samples].clone() for t, n in self.partial_noises.items()},
            seed=self.seed,
            device=self.device
        )
    
    def validate_batch_size(self, expected_batch_size: int) -> None:
        """
        Validate that this cache matches the expected batch size.
        
        Raises:
            ValueError if batch sizes don't match
        """
        if self.batch_size != expected_batch_size:
            raise ValueError(
                f"Noise cache batch_size mismatch: cache has {self.batch_size} samples, "
                f"but {expected_batch_size} were requested. "
                f"Either regenerate cache with --n_samples={expected_batch_size}, "
                f"or use cache.slice({expected_batch_size}) if cache has more samples."
            )
    
    @staticmethod
    def combine(caches: list) -> 'DiffusionNoiseCache':
        """
        Combine multiple noise caches into one for batch processing.
        
        Used when processing multiple scenes in a single batch - each scene
        has its own cache, and we concatenate them along the batch dimension.
        
        Args:
            caches: List of DiffusionNoiseCache objects (all must have same structure)
        
        Returns:
            New DiffusionNoiseCache with concatenated tensors
        
        Example:
            # Each cache has (n_samples, num_points, point_dim)
            # Combined cache has (num_scenes * n_samples, num_points, point_dim)
            combined = DiffusionNoiseCache.combine([cache1, cache2, cache3])
        """
        if not caches:
            raise ValueError("Cannot combine empty list of caches")
        
        if len(caches) == 1:
            return caches[0].clone()
        
        # Verify all caches have same structure
        ref = caches[0]
        for i, c in enumerate(caches[1:], 1):
            if c.batch_size != ref.batch_size:
                raise ValueError(f"Cache {i} has batch_size {c.batch_size}, expected {ref.batch_size}")
        
        # Concatenate initial noise: (B1, N, D) + (B2, N, D) + ... -> (B1+B2+..., N, D)
        initial_noise = torch.cat([c.initial_noise for c in caches], dim=0)
        
        # Concatenate timestep noises
        timestep_noises = {}
        for t in ref.timestep_noises.keys():
            timestep_noises[t] = torch.cat([c.timestep_noises[t] for c in caches], dim=0)
        
        # Concatenate partial noises
        partial_noises = {}
        if ref.partial_noises:
            for t in ref.partial_noises.keys():
                partial_noises[t] = torch.cat([c.partial_noises[t] for c in caches], dim=0)
        
        return DiffusionNoiseCache(
            initial_noise=initial_noise,
            timestep_noises=timestep_noises,
            partial_noises=partial_noises,
            seed=-1,  # Combined cache doesn't have a single seed
            device=ref.device
        )

