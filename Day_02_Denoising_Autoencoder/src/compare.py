"""
Comparison and limitations analysis for Day 2: Denoising Autoencoder
"""

import html
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

from .metrics import MetricsCalculator
from .utils import console, save_image_grid


class DAELimitationsAnalyzer:
    """Analyze limitations of denoising autoencoders vs generative models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        output_dir: Union[str, Path]
    ):
        self.model = model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.metrics_calc = MetricsCalculator(device)
        
        # Create output directories
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "grids").mkdir(parents=True, exist_ok=True)
    
    def analyze_over_smoothing(
        self,
        clean_images: torch.Tensor,
        reconstructed_images: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Analyze over-smoothing by comparing high-frequency content.
        
        Args:
            clean_images: Original clean images [B, C, H, W]
            reconstructed_images: Model reconstructions [B, C, H, W]
            num_samples: Number of samples to analyze
        
        Returns:
            Dictionary with smoothing analysis results
        """
        # Ensure tensors are on the correct device
        clean_images = clean_images.to(self.device)
        reconstructed_images = reconstructed_images.to(self.device)
        
        num_samples = min(num_samples, clean_images.size(0))
        
        clean_hf_energy = []
        recon_hf_energy = []
        gradient_magnitudes_clean = []
        gradient_magnitudes_recon = []
        
        for i in range(num_samples):
            clean = clean_images[i:i+1]
            recon = reconstructed_images[i:i+1]
            
            # High-frequency energy via Laplacian
            laplacian_kernel = torch.tensor([
                [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]]
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Handle multi-channel images
            if clean.size(1) > 1:
                # Convert to grayscale for analysis
                clean_gray = 0.299 * clean[:, 0:1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
                recon_gray = 0.299 * recon[:, 0:1] + 0.587 * recon[:, 1:2] + 0.114 * recon[:, 2:3]
            else:
                clean_gray = clean
                recon_gray = recon
            
            # Apply Laplacian
            clean_hf = F.conv2d(clean_gray, laplacian_kernel, padding=1)
            recon_hf = F.conv2d(recon_gray, laplacian_kernel, padding=1)
            
            # Compute energy (RMS of high-frequency components)
            clean_energy = torch.sqrt(torch.mean(clean_hf ** 2)).item()
            recon_energy = torch.sqrt(torch.mean(recon_hf ** 2)).item()
            
            clean_hf_energy.append(clean_energy)
            recon_hf_energy.append(recon_energy)
            
            # Gradient magnitude analysis - ensure matching dimensions
            grad_x_clean = clean_gray[:, :, :, 1:] - clean_gray[:, :, :, :-1]
            grad_y_clean = clean_gray[:, :, 1:, :] - clean_gray[:, :, :-1, :]
            
            # Take intersection to ensure same spatial dimensions
            grad_x_clean_crop = grad_x_clean[:, :, :-1, :]  # Remove last row
            grad_y_clean_crop = grad_y_clean[:, :, :, :-1]  # Remove last column
            grad_mag_clean = torch.sqrt(grad_x_clean_crop**2 + grad_y_clean_crop**2).mean().item()
            
            grad_x_recon = recon_gray[:, :, :, 1:] - recon_gray[:, :, :, :-1]
            grad_y_recon = recon_gray[:, :, 1:, :] - recon_gray[:, :, :-1, :]
            
            # Take intersection to ensure same spatial dimensions
            grad_x_recon_crop = grad_x_recon[:, :, :-1, :]  # Remove last row
            grad_y_recon_crop = grad_y_recon[:, :, :, :-1]  # Remove last column
            grad_mag_recon = torch.sqrt(grad_x_recon_crop**2 + grad_y_recon_crop**2).mean().item()
            
            gradient_magnitudes_clean.append(grad_mag_clean)
            gradient_magnitudes_recon.append(grad_mag_recon)
        
        # Statistical analysis
        hf_ratio = np.mean(recon_hf_energy) / np.mean(clean_hf_energy) if np.mean(clean_hf_energy) > 0 else 0
        grad_ratio = np.mean(gradient_magnitudes_recon) / np.mean(gradient_magnitudes_clean) if np.mean(gradient_magnitudes_clean) > 0 else 0
        
        # Statistical significance test
        _, hf_pvalue = stats.ttest_rel(clean_hf_energy, recon_hf_energy)
        _, grad_pvalue = stats.ttest_rel(gradient_magnitudes_clean, gradient_magnitudes_recon)
        
        results = {
            'hf_energy_ratio': hf_ratio,
            'gradient_ratio': grad_ratio,
            'hf_energy_clean_mean': np.mean(clean_hf_energy),
            'hf_energy_recon_mean': np.mean(recon_hf_energy),
            'gradient_clean_mean': np.mean(gradient_magnitudes_clean),
            'gradient_recon_mean': np.mean(gradient_magnitudes_recon),
            'hf_pvalue': hf_pvalue,
            'gradient_pvalue': grad_pvalue,
            'smoothing_score': 1 - min(hf_ratio, grad_ratio)  # Higher = more smoothing
        }
        
        return results
    
    def analyze_diversity_collapse(
        self,
        clean_image: torch.Tensor,
        num_noise_samples: int = 20,
        noise_sigma: float = 0.3
    ) -> Dict[str, float]:
        """
        Analyze diversity collapse by testing multiple noisy versions of same image.
        DAEs should produce similar outputs for different noise realizations.
        
        Args:
            clean_image: Single clean image [C, H, W]
            num_noise_samples: Number of different noise realizations
            noise_sigma: Noise level to use
        
        Returns:
            Dictionary with diversity analysis results
        """
        clean_batch = clean_image.unsqueeze(0).to(self.device)
        reconstructions = []
        
        with torch.no_grad():
            for i in range(num_noise_samples):
                # Generate different noise realizations
                torch.manual_seed(i)  # Different seed for each sample
                noise = torch.randn_like(clean_batch) * noise_sigma
                noisy = torch.clamp(clean_batch + noise, 0, 1)
                
                recon = self.model(noisy)
                reconstructions.append(recon.cpu())
        
        # Stack all reconstructions
        all_recons = torch.cat(reconstructions, dim=0)  # [N, C, H, W]
        
        # Compute pairwise distances
        distances = []
        for i in range(num_noise_samples):
            for j in range(i + 1, num_noise_samples):
                dist = F.mse_loss(all_recons[i], all_recons[j]).item()
                distances.append(dist)
        
        # Compute statistics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        
        # Normalized diversity score (lower = less diverse)
        diversity_score = mean_distance / (noise_sigma ** 2)  # Normalize by noise level
        
        results = {
            'mean_pairwise_distance': mean_distance,
            'std_pairwise_distance': std_distance,
            'max_pairwise_distance': max_distance,
            'min_pairwise_distance': min_distance,
            'diversity_score': diversity_score,
            'num_samples': num_noise_samples,
            'noise_sigma': noise_sigma,
            'collapse_indicator': 1 - min(diversity_score, 1.0)  # Higher = more collapse
        }
        
        return results
    
    def create_diversity_visualization(
        self,
        clean_image: torch.Tensor,
        noise_sigma: float = 0.3,
        num_samples: int = 8
    ) -> None:
        """Create visualization showing diversity collapse."""
        clean_batch = clean_image.unsqueeze(0).to(self.device)
        
        viz_data = [clean_image.cpu()]  # Start with clean image (ensure CPU)
        
        with torch.no_grad():
            for i in range(num_samples):
                torch.manual_seed(i)
                noise = torch.randn_like(clean_batch) * noise_sigma
                noisy = torch.clamp(clean_batch + noise, 0, 1)
                recon = self.model(noisy)
                viz_data.append(recon.squeeze(0).cpu())
        
        # Save visualization
        viz_tensor = torch.stack(viz_data)
        save_path = self.output_dir / "grids" / "diversity_collapse_demo.png"
        save_image_grid(viz_tensor, save_path, nrow=num_samples + 1)
        
        console.print(f"[blue]Saved diversity visualization to {save_path}[/blue]")
    
    def compare_frequency_content(
        self,
        clean_images: torch.Tensor,
        recon_images: torch.Tensor,
        save_plot: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compare frequency content between clean and reconstructed images using FFT.
        
        Args:
            clean_images: Clean images [B, C, H, W]
            recon_images: Reconstructed images [B, C, H, W]
            save_plot: Whether to save frequency comparison plot
        
        Returns:
            Dictionary with frequency analysis results
        """
        # Ensure tensors are on the correct device
        clean_images = clean_images.to(self.device)
        recon_images = recon_images.to(self.device)
        def compute_frequency_spectrum(images):
            """Compute average frequency spectrum."""
            spectra = []
            
            for i in range(images.size(0)):
                img = images[i]
                
                # Convert to grayscale if needed
                if img.size(0) > 1:
                    img_gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                else:
                    img_gray = img[0]
                
                # Compute 2D FFT
                fft = torch.fft.fft2(img_gray)
                spectrum = torch.abs(fft)
                
                # Shift zero frequency to center
                spectrum = torch.fft.fftshift(spectrum)
                spectra.append(spectrum.cpu().numpy())
            
            return np.mean(spectra, axis=0)
        
        clean_spectrum = compute_frequency_spectrum(clean_images)
        recon_spectrum = compute_frequency_spectrum(recon_images)
        
        # Compute radial average
        def radial_profile(spectrum):
            h, w = spectrum.shape
            center = (h // 2, w // 2)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Bin by radius
            r_max = int(r.max())
            profile = np.zeros(r_max)
            
            for radius in range(r_max):
                mask = (r >= radius) & (r < radius + 1)
                if np.any(mask):
                    profile[radius] = spectrum[mask].mean()
            
            return profile
        
        clean_profile = radial_profile(clean_spectrum)
        recon_profile = radial_profile(recon_spectrum)
        
        # Create frequency comparison plot
        if save_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # 2D spectra
            ax1.imshow(np.log(clean_spectrum + 1), cmap='viridis')
            ax1.set_title('Clean Images\nFrequency Spectrum')
            ax1.axis('off')
            
            ax2.imshow(np.log(recon_spectrum + 1), cmap='viridis')
            ax2.set_title('Reconstructed Images\nFrequency Spectrum')
            ax2.axis('off')
            
            # Radial profiles
            frequencies = np.arange(len(clean_profile))
            ax3.semilogy(frequencies, clean_profile, 'b-', label='Clean', linewidth=2)
            ax3.semilogy(frequencies, recon_profile, 'r-', label='Reconstructed', linewidth=2)
            ax3.set_xlabel('Spatial Frequency')
            ax3.set_ylabel('Magnitude')
            ax3.set_title('Radial Frequency Profile')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.output_dir / "reports" / "frequency_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print(f"[blue]Saved frequency analysis to {save_path}[/blue]")
        
        return {
            'clean_spectrum': clean_spectrum,
            'recon_spectrum': recon_spectrum,
            'clean_profile': clean_profile,
            'recon_profile': recon_profile,
            'high_freq_ratio': np.sum(recon_profile[-len(recon_profile)//4:]) / np.sum(clean_profile[-len(clean_profile)//4:])
        }
    
    def generate_limitations_report(
        self,
        test_loader: DataLoader,
        max_batches: int = 10
    ) -> Dict:
        """Generate comprehensive limitations analysis report."""
        console.print("[bold blue]Analyzing DAE limitations vs generative models[/bold blue]")
        
        # Collect data
        all_clean = []
        all_recon = []
        
        with torch.no_grad():
            for batch_idx, (clean, noisy, _) in enumerate(test_loader):
                if batch_idx >= max_batches:
                    break
                
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                recon = self.model(noisy)
                
                all_clean.append(clean)
                all_recon.append(recon)
        
        if not all_clean:
            return {}
        
        clean_tensor = torch.cat(all_clean, dim=0)
        recon_tensor = torch.cat(all_recon, dim=0)
        
        # 1. Over-smoothing analysis
        console.print("1. Analyzing over-smoothing...")
        smoothing_results = self.analyze_over_smoothing(clean_tensor, recon_tensor)
        
        # 2. Diversity collapse analysis
        console.print("2. Analyzing diversity collapse...")
        sample_image = clean_tensor[0]  # Use first image
        diversity_results = self.analyze_diversity_collapse(sample_image)
        
        # 3. Frequency content analysis
        console.print("3. Analyzing frequency content...")
        frequency_results = self.compare_frequency_content(
            clean_tensor[:50], recon_tensor[:50]  # Limit for efficiency
        )
        
        # 4. Create visualizations
        console.print("4. Creating visualizations...")
        self.create_diversity_visualization(sample_image)
        
        # Compile results
        results = {
            'over_smoothing': smoothing_results,
            'diversity_collapse': diversity_results,
            'frequency_analysis': {
                k: v for k, v in frequency_results.items()
                if k not in ['clean_spectrum', 'recon_spectrum']  # Exclude large arrays
            },
            'summary': {
                'smoothing_severity': smoothing_results['smoothing_score'],
                'diversity_collapse_severity': diversity_results['collapse_indicator'],
                'high_freq_preservation': frequency_results['high_freq_ratio'],
                'overall_limitation_score': (
                    smoothing_results['smoothing_score'] +
                    diversity_results['collapse_indicator'] +
                    (1 - frequency_results['high_freq_ratio'])
                ) / 3
            }
        }
        
        # Generate HTML report
        self._generate_html_report(results)
        
        return results
    
    def _generate_html_report(self, results: Dict) -> None:
        """Generate HTML report of limitations analysis."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Denoising Autoencoder Limitations Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ margin: 10px 0; }}
                .score {{ font-weight: bold; color: #d63031; }}
                .good {{ color: #00b894; }}
                .warning {{ color: #fdcb6e; }}
                .bad {{ color: #d63031; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Denoising Autoencoder Limitations Analysis</h1>
                <p>Analysis of DAE limitations compared to generative models</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>Denoising Autoencoders (DAEs) are reconstruction-based models that learn to map noisy inputs to clean outputs. 
                Unlike generative models, DAEs have fundamental limitations in diversity and detail preservation.</p>
                
                <table>
                    <tr>
                        <th>Limitation</th>
                        <th>Score (0-1, higher = worse)</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Over-smoothing</td>
                        <td class="score">{results['summary']['smoothing_severity']:.3f}</td>
                        <td>Tendency to blur fine details</td>
                    </tr>
                    <tr>
                        <td>Diversity Collapse</td>
                        <td class="score">{results['summary']['diversity_collapse_severity']:.3f}</td>
                        <td>Similar outputs from different noisy inputs</td>
                    </tr>
                    <tr>
                        <td>High-Freq Loss</td>
                        <td class="score">{1 - results['summary']['high_freq_preservation']:.3f}</td>
                        <td>Loss of high-frequency image content</td>
                    </tr>
                    <tr>
                        <td><strong>Overall</strong></td>
                        <td class="score"><strong>{results['summary']['overall_limitation_score']:.3f}</strong></td>
                        <td><strong>Combined limitation severity</strong></td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>1. Over-Smoothing Analysis</h2>
                <p>DAEs tend to over-smooth images by reducing high-frequency content.</p>
                <div class="metric">High-frequency energy ratio: <span class="score">{results['over_smoothing']['hf_energy_ratio']:.3f}</span></div>
                <div class="metric">Gradient magnitude ratio: <span class="score">{results['over_smoothing']['gradient_ratio']:.3f}</span></div>
                <p><strong>Interpretation:</strong> Values < 1.0 indicate smoothing (loss of detail). 
                Typical values for good reconstruction: 0.7-0.9.</p>
            </div>
            
            <div class="section">
                <h2>2. Diversity Collapse Analysis</h2>
                <p>DAEs produce similar outputs when given different noisy versions of the same image.</p>
                <div class="metric">Mean pairwise distance: <span class="score">{results['diversity_collapse']['mean_pairwise_distance']:.6f}</span></div>
                <div class="metric">Diversity score: <span class="score">{results['diversity_collapse']['diversity_score']:.3f}</span></div>
                <p><strong>Interpretation:</strong> Low diversity scores indicate mode collapse. 
                Generative models would show higher diversity.</p>
            </div>
            
            <div class="section">
                <h2>3. Key Limitations vs Generative Models</h2>
                <ul>
                    <li><strong>No Distribution Modeling:</strong> DAEs learn mappings, not data distributions</li>
                    <li><strong>Limited Diversity:</strong> Cannot generate novel variations</li>
                    <li><strong>Over-smoothing:</strong> Tend to average out fine details</li>
                    <li><strong>No Sampling:</strong> Cannot generate new samples from noise</li>
                    <li><strong>Mode Averaging:</strong> May blend different modes rather than choosing</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>4. When DAEs Fall Short</h2>
                <ul>
                    <li>High noise levels (σ > 0.5)</li>
                    <li>Images with fine textures or details</li>
                    <li>Multi-modal reconstruction scenarios</li>
                    <li>Creative/generative applications</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>5. Conclusion</h2>
                <p>While DAEs are effective for denoising tasks, they have fundamental limitations compared to 
                generative models like diffusion models. They excel at reconstruction but cannot model the 
                underlying data distribution, leading to over-smoothing and lack of diversity.</p>
                
                <p><strong>Next Steps:</strong> Understanding these limitations motivates the study of generative 
                models (VAEs, GANs, Diffusion Models) that can model data distributions and generate diverse, 
                high-quality samples.</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = self.output_dir / "reports" / "limitations_analysis.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        console.print(f"[green]Generated limitations report: {report_path}[/green]")


def analyze_limitations(
    model: torch.nn.Module,
    test_loader: DataLoader,
    config,
    device: torch.device
) -> Dict:
    """
    Main function to analyze DAE limitations.
    
    Args:
        model: Trained DAE model
        test_loader: Test data loader
        config: Configuration object
        device: Computation device
    
    Returns:
        Analysis results dictionary
    """
    output_dir = Path(config.log.out_dir)
    analyzer = DAELimitationsAnalyzer(model, device, output_dir)
    
    results = analyzer.generate_limitations_report(test_loader)
    
    # Print summary
    if results:
        summary = results['summary']
        console.print(f"[yellow]Limitations Summary:[/yellow]")
        console.print(f"  Over-smoothing severity: {summary['smoothing_severity']:.3f}")
        console.print(f"  Diversity collapse severity: {summary['diversity_collapse_severity']:.3f}")
        console.print(f"  High-frequency preservation: {summary['high_freq_preservation']:.3f}")
        console.print(f"  Overall limitation score: {summary['overall_limitation_score']:.3f}")
    
    return results