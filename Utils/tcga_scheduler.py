import numpy as np
import matplotlib.pyplot as plt
import torch

class TCGARatioScheduler:
    """
    Scheduler for dynamic adjustment of TCGA ratio during training.
    This allows the model to benefit from TCGA's long-range modeling early in training
    and then gradually focus more on standard attention patterns later.
    """
    def __init__(self, initial_ratio=0.8, final_ratio=0.3, total_iterations=10000, schedule_type='cosine'):
        """
        Args:
            initial_ratio: Starting TCGA ratio (higher means more TCGA influence)
            final_ratio: Final TCGA ratio to reach at the end of training
            total_iterations: Total number of training iterations
            schedule_type: Type of schedule ('cosine', 'linear', or 'step')
        """
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_iterations = total_iterations
        self.schedule_type = schedule_type
        
    def get_ratio(self, current_iteration):
        """
        Calculate the TCGA ratio for the current training iteration.
        """
        progress = min(current_iteration / self.total_iterations, 1.0)
        
        if self.schedule_type == 'cosine':
            # Cosine annealing schedule
            ratio = self.final_ratio + 0.5 * (self.initial_ratio - self.final_ratio) * \
                    (1 + np.cos(np.pi * progress))
                    
        elif self.schedule_type == 'linear':
            # Linear decay
            ratio = self.initial_ratio - progress * (self.initial_ratio - self.final_ratio)
            
        elif self.schedule_type == 'step':
            # Step schedule (high ratio until halfway, then drop)
            if progress < 0.5:
                ratio = self.initial_ratio
            else:
                ratio = self.final_ratio
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
            
        return ratio
    
    def plot_schedule(self, save_path=None):
        """
        Visualize the TCGA ratio schedule.
        """
        iterations = np.arange(0, self.total_iterations + 1)
        ratios = [self.get_ratio(it) for it in iterations]
        
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, ratios)
        plt.title(f'TCGA Ratio Schedule ({self.schedule_type})')
        plt.xlabel('Iterations')
        plt.ylabel('TCGA Ratio')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    scheduler = TCGARatioScheduler(initial_ratio=0.8, final_ratio=0.3, total_iterations=10000)
    scheduler.plot_schedule("tcga_schedule.png")
