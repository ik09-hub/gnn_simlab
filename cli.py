import yaml
from pipeline.simulate import run_simulation
from pipeline.train import run_training
from pipeline.evaluate import run_evaluation



def main():
    spec_path = "specs/example_fraud.yaml"

    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)
    

    run_dir = run_simulation(spec)

    model_path = run_training(run_dir, spec)

    evaluate = run_evaluation(run_dir, spec)
    print(f"Simulation complete. Outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()

