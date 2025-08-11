"""Results analysis and visualization tools"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ResultsAnalyzer:
    """Analyzes and visualizes attack test results"""
    
    def __init__(self, results_dir: str = "./results"):
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path(results_dir)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Load all available results
        self.results_data = self._load_all_results()
        
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files from the results directory"""
        results = []
        
        if not self.results_dir.exists():
            self.logger.warning(f"Results directory {self.results_dir} does not exist")
            return results
        
        # Load evaluation files
        for file_path in self.results_dir.glob("evaluation_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = str(file_path)
                    results.append(data)
                    
                self.logger.info(f"Loaded results from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
        
        # Also load attack results files
        for file_path in self.results_dir.glob("attack_results_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = str(file_path)
                    data['result_type'] = 'attack_results'
                    results.append(data)
                    
                self.logger.info(f"Loaded attack results from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
        
        return results
    
    def create_robustness_comparison_chart(self, save_path: Optional[str] = None) -> str:
        """Create robustness comparison chart across models"""
        
        # Extract robustness data
        model_robustness = {}
        
        for result in self.results_data:
            if 'model_evaluations' in result:
                for model_name, evaluation in result['model_evaluations'].items():
                    if model_name not in model_robustness:
                        model_robustness[model_name] = []
                    
                    robustness_score = evaluation['robustness_score']['overall_score']
                    model_robustness[model_name].append(robustness_score)
        
        if not model_robustness:
            self.logger.warning("No robustness data found")
            return ""
        
        # Calculate averages and create chart
        model_names = list(model_robustness.keys())
        avg_scores = [np.mean(scores) for scores in model_robustness.values()]
        std_scores = [np.std(scores) if len(scores) > 1 else 0 for scores in model_robustness.values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(model_names, avg_scores, yerr=std_scores, capsize=5)
        
        # Customize chart
        ax.set_xlabel('Model')
        ax.set_ylabel('Robustness Score')
        ax.set_title('Model Robustness Comparison')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(model_names) > 6:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = self.results_dir / f"robustness_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Robustness comparison chart saved to {save_path}")
        return str(save_path)
    
    def create_attack_effectiveness_heatmap(self, save_path: Optional[str] = None) -> str:
        """Create heatmap showing attack effectiveness by type and model"""
        
        # Extract attack effectiveness data
        effectiveness_data = defaultdict(lambda: defaultdict(list))
        
        for result in self.results_data:
            if 'model_evaluations' in result:
                for model_name, evaluation in result['model_evaluations'].items():
                    vuln_patterns = evaluation.get('vulnerability_patterns', {})
                    
                    if 'by_attack_type' in vuln_patterns:
                        for attack_type, stats in vuln_patterns['by_attack_type'].items():
                            success_rate = stats.get('success_rate', 0)
                            effectiveness_data[attack_type][model_name].append(success_rate)
        
        if not effectiveness_data:
            self.logger.warning("No attack effectiveness data found")
            return ""
        
        # Create matrix for heatmap
        attack_types = list(effectiveness_data.keys())
        model_names = set()
        for attack_data in effectiveness_data.values():
            model_names.update(attack_data.keys())
        model_names = sorted(list(model_names))
        
        matrix = np.zeros((len(attack_types), len(model_names)))
        
        for i, attack_type in enumerate(attack_types):
            for j, model_name in enumerate(model_names):
                if model_name in effectiveness_data[attack_type]:
                    scores = effectiveness_data[attack_type][model_name]
                    matrix[i, j] = np.mean(scores) if scores else 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.3f', 
                   xticklabels=model_names,
                   yticklabels=attack_types,
                   cmap='Reds',
                   vmin=0,
                   vmax=1,
                   ax=ax)
        
        ax.set_title('Attack Success Rate by Type and Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Attack Type')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save heatmap
        if save_path is None:
            save_path = self.results_dir / f"attack_effectiveness_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Attack effectiveness heatmap saved to {save_path}")
        return str(save_path)
    
    def create_vulnerability_distribution_chart(self, save_path: Optional[str] = None) -> str:
        """Create chart showing distribution of vulnerabilities across attack types"""
        
        # Aggregate vulnerability data
        attack_type_vulnerabilities = defaultdict(list)
        
        for result in self.results_data:
            if 'attack_effectiveness' in result:
                by_type = result['attack_effectiveness'].get('by_attack_type', {})
                for attack_type, stats in by_type.items():
                    success_rate = stats.get('success_rate', 0)
                    attack_type_vulnerabilities[attack_type].append(success_rate)
        
        if not attack_type_vulnerabilities:
            self.logger.warning("No vulnerability distribution data found")
            return ""
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attack_types = list(attack_type_vulnerabilities.keys())
        vulnerability_data = [attack_type_vulnerabilities[at] for at in attack_types]
        
        box_plot = ax.boxplot(vulnerability_data, 
                             labels=attack_types,
                             patch_artist=True)
        
        # Customize colors
        colors = sns.color_palette("husl", len(attack_types))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Success Rate')
        ax.set_title('Vulnerability Distribution by Attack Type')
        ax.set_ylim(0, 1)
        
        # Add mean markers
        means = [np.mean(data) for data in vulnerability_data]
        ax.scatter(range(1, len(means) + 1), means, 
                  marker='D', s=50, color='red', zorder=3, label='Mean')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = self.results_dir / f"vulnerability_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Vulnerability distribution chart saved to {save_path}")
        return str(save_path)
    
    def create_performance_correlation_chart(self, save_path: Optional[str] = None) -> str:
        """Create scatter plot showing correlation between model size and robustness"""
        
        # Extract model performance and metadata
        model_data = []
        
        for result in self.results_data:
            if 'model_evaluations' in result:
                for model_name, evaluation in result['model_evaluations'].items():
                    robustness = evaluation['robustness_score']['overall_score']
                    
                    # Extract parameter count from model name (heuristic)
                    param_count = self._extract_param_count_numeric(model_name)
                    
                    if param_count is not None:
                        model_data.append({
                            'model': model_name,
                            'robustness': robustness,
                            'param_count': param_count,
                            'size_category': self._categorize_model_size(param_count)
                        })
        
        if not model_data:
            self.logger.warning("Insufficient data for performance correlation")
            return ""
        
        # Create scatter plot
        df = pd.DataFrame(model_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by size category
        size_categories = df['size_category'].unique()
        colors = sns.color_palette("husl", len(size_categories))
        color_map = dict(zip(size_categories, colors))
        
        for category in size_categories:
            category_data = df[df['size_category'] == category]
            ax.scatter(category_data['param_count'], 
                      category_data['robustness'],
                      c=[color_map[category]], 
                      label=category,
                      s=60,
                      alpha=0.7)
        
        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df['param_count'], df['robustness'], 1)
            p = np.poly1d(z)
            ax.plot(df['param_count'].sort_values(), 
                   p(df['param_count'].sort_values()), 
                   "r--", alpha=0.8, linewidth=1)
        
        ax.set_xlabel('Parameter Count (Billions)')
        ax.set_ylabel('Robustness Score')
        ax.set_title('Model Size vs. Robustness Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add model labels
        for _, row in df.iterrows():
            ax.annotate(row['model'].split(':')[0], 
                       (row['param_count'], row['robustness']),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.7)
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = self.results_dir / f"performance_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance correlation chart saved to {save_path}")
        return str(save_path)
    
    def _extract_param_count_numeric(self, model_name: str) -> Optional[float]:
        """Extract numeric parameter count from model name"""
        import re
        
        # Patterns for parameter counts
        patterns = [
            r'(\d+\.?\d*)[bB]',  # 7b, 3.8b, 70b
            r':(\d+\.?\d*)[bB]', # :7b, :3.8b
            r'(\d+)x(\d+)[bB]',  # 8x7b (mixtral)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                if 'x' in pattern:
                    # Handle MoE models (approximate total)
                    experts = float(match.group(1))
                    size = float(match.group(2))
                    return experts * size
                else:
                    return float(match.group(1))
        
        return None
    
    def _categorize_model_size(self, param_count: float) -> str:
        """Categorize model by parameter count"""
        if param_count < 7:
            return "Small (<7B)"
        elif param_count < 15:
            return "Medium (7-15B)"
        elif param_count < 35:
            return "Large (15-35B)"
        else:
            return "Very Large (35B+)"
    
    def generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistical summary of results"""
        
        summary = {
            "data_overview": {},
            "model_statistics": {},
            "attack_statistics": {},
            "performance_insights": []
        }
        
        # Data overview
        summary["data_overview"] = {
            "total_result_files": len(self.results_data),
            "date_range": self._get_date_range(),
            "models_tested": self._get_unique_models(),
            "attack_types_tested": self._get_unique_attack_types()
        }
        
        # Model statistics
        model_robustness = self._aggregate_model_robustness()
        if model_robustness:
            summary["model_statistics"] = {
                "most_robust_model": max(model_robustness.keys(), key=lambda k: np.mean(model_robustness[k])),
                "least_robust_model": min(model_robustness.keys(), key=lambda k: np.mean(model_robustness[k])),
                "robustness_statistics": {
                    model: {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": min(scores),
                        "max": max(scores)
                    }
                    for model, scores in model_robustness.items()
                }
            }
        
        # Attack statistics
        attack_effectiveness = self._aggregate_attack_effectiveness()
        if attack_effectiveness:
            summary["attack_statistics"] = {
                "most_effective_attack": max(attack_effectiveness.keys(), key=lambda k: np.mean(attack_effectiveness[k])),
                "least_effective_attack": min(attack_effectiveness.keys(), key=lambda k: np.mean(attack_effectiveness[k])),
                "effectiveness_by_type": {
                    attack_type: {
                        "mean_success_rate": np.mean(rates),
                        "std": np.std(rates),
                        "attempts": len(rates)
                    }
                    for attack_type, rates in attack_effectiveness.items()
                }
            }
        
        # Generate insights
        summary["performance_insights"] = self._generate_statistical_insights(
            model_robustness, attack_effectiveness
        )
        
        return summary
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get date range of results"""
        timestamps = []
        for result in self.results_data:
            if 'timestamp' in result:
                timestamps.append(result['timestamp'])
        
        if timestamps:
            return {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }
        return {"earliest": "unknown", "latest": "unknown"}
    
    def _get_unique_models(self) -> List[str]:
        """Get list of unique models tested"""
        models = set()
        for result in self.results_data:
            if 'model_evaluations' in result:
                models.update(result['model_evaluations'].keys())
        return sorted(list(models))
    
    def _get_unique_attack_types(self) -> List[str]:
        """Get list of unique attack types tested"""
        attack_types = set()
        for result in self.results_data:
            if 'attack_effectiveness' in result:
                by_type = result['attack_effectiveness'].get('by_attack_type', {})
                attack_types.update(by_type.keys())
        return sorted(list(attack_types))
    
    def _aggregate_model_robustness(self) -> Dict[str, List[float]]:
        """Aggregate robustness scores by model"""
        model_scores = defaultdict(list)
        
        for result in self.results_data:
            if 'model_evaluations' in result:
                for model_name, evaluation in result['model_evaluations'].items():
                    score = evaluation['robustness_score']['overall_score']
                    model_scores[model_name].append(score)
        
        return dict(model_scores)
    
    def _aggregate_attack_effectiveness(self) -> Dict[str, List[float]]:
        """Aggregate attack effectiveness by type"""
        attack_rates = defaultdict(list)
        
        for result in self.results_data:
            if 'attack_effectiveness' in result:
                by_type = result['attack_effectiveness'].get('by_attack_type', {})
                for attack_type, stats in by_type.items():
                    success_rate = stats.get('success_rate', 0)
                    attack_rates[attack_type].append(success_rate)
        
        return dict(attack_rates)
    
    def _generate_statistical_insights(
        self,
        model_robustness: Dict[str, List[float]],
        attack_effectiveness: Dict[str, List[float]]
    ) -> List[str]:
        """Generate statistical insights from aggregated data"""
        
        insights = []
        
        # Model insights
        if model_robustness:
            avg_robustness = {model: np.mean(scores) for model, scores in model_robustness.items()}
            robustness_variance = np.var(list(avg_robustness.values()))
            
            if robustness_variance < 0.01:
                insights.append("Models show similar robustness levels (low variance)")
            else:
                insights.append("Significant robustness differences between models")
            
            # Model size correlation
            size_robustness = []
            for model, avg_score in avg_robustness.items():
                param_count = self._extract_param_count_numeric(model)
                if param_count:
                    size_robustness.append((param_count, avg_score))
            
            if len(size_robustness) > 3:
                correlation = np.corrcoef([x[0] for x in size_robustness], 
                                        [x[1] for x in size_robustness])[0,1]
                if correlation > 0.5:
                    insights.append("Larger models tend to be more robust")
                elif correlation < -0.5:
                    insights.append("Smaller models tend to be more robust")
                else:
                    insights.append("No clear correlation between model size and robustness")
        
        # Attack insights
        if attack_effectiveness:
            avg_effectiveness = {attack: np.mean(rates) for attack, rates in attack_effectiveness.items()}
            
            most_effective = max(avg_effectiveness.keys(), key=lambda k: avg_effectiveness[k])
            least_effective = min(avg_effectiveness.keys(), key=lambda k: avg_effectiveness[k])
            
            insights.append(f"Most effective attack type: {most_effective} ({avg_effectiveness[most_effective]:.1%} success rate)")
            insights.append(f"Least effective attack type: {least_effective} ({avg_effectiveness[least_effective]:.1%} success rate)")
        
        return insights
    
    def export_analysis_report(self, output_path: Optional[str] = None) -> str:
        """Export comprehensive analysis report"""
        
        if output_path is None:
            output_path = self.results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Generate all visualizations
        chart_paths = []
        try:
            chart_paths.append(self.create_robustness_comparison_chart())
            chart_paths.append(self.create_attack_effectiveness_heatmap())
            chart_paths.append(self.create_vulnerability_distribution_chart())
            chart_paths.append(self.create_performance_correlation_chart())
        except Exception as e:
            self.logger.error(f"Error creating charts: {e}")
        
        # Generate statistical summary
        statistical_summary = self.generate_statistical_summary()
        
        # Compile full report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_sources": [result['source_file'] for result in self.results_data],
            "visualizations_created": chart_paths,
            "statistical_summary": statistical_summary,
            "raw_data_summary": {
                "total_evaluations": len(self.results_data),
                "models_analyzed": len(self._get_unique_models()),
                "attack_types_analyzed": len(self._get_unique_attack_types())
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report saved to {output_path}")
        return str(output_path)