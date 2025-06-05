#!/usr/bin/env python3
"""
Temperature Analysis Experiment
This script runs knowledge distillation with different temperature values
to analyze the effect of temperature on student performance.
"""

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_temperature_experiment():
    """Run knowledge distillation with different temperature values"""
    temperatures = [1, 2, 3, 4, 5, 6, 8, 10]
    results = {}

    print("Running Temperature Analysis Experiment...")
    print("=" * 50)

    for temp in temperatures:
        print(f"\nTesting Temperature: {temp}")
        print("-" * 30)

        # Run knowledge distillation with current temperature
        cmd = [
            "python", "main.py",
            "--mode", "distill",
            "--temperature", str(temp),
            "--epochs", "15",
            "--alpha", "0.3"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            # Parse the output to extract final accuracy
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Student Accuracy:" in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results[temp] = accuracy
                    print(f"Temperature {temp}: {accuracy:.2f}%")
                    break
            else:
                print(f"Could not parse accuracy for temperature {temp}")
                results[temp] = 0.0

        except subprocess.TimeoutExpired:
            print(f"Training with temperature {temp} timed out")
            results[temp] = 0.0
        except Exception as e:
            print(f"Error with temperature {temp}: {e}")
            results[temp] = 0.0

    # Save results
    with open('temperature_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot results
    plot_temperature_results(results)

    return results

def plot_temperature_results(results):
    """Plot temperature vs accuracy results"""
    temps = list(results.keys())
    accs = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(temps, accs, 'bo-', linewidth=2, markersize=8)
    plt.title('Effect of Temperature on Knowledge Distillation Performance', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Student Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Annotate best result
    best_temp = max(results, key=results.get)
    best_acc = results[best_temp]
    plt.annotate(f'Best: T={best_temp}, Acc={best_acc:.2f}%', 
                xy=(best_temp, best_acc), xytext=(best_temp+1, best_acc+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nBest temperature: {best_temp} (Accuracy: {best_acc:.2f}%)")

if __name__ == "__main__":
    results = run_temperature_experiment()
    print("\nTemperature experiment completed!")
    print("Results saved to 'temperature_results.json'")
    print("Plot saved to 'temperature_analysis.png'")
