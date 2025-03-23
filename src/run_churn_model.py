from bank_churn_predictor import BankChurnPredictor

# Create the churn predictor with your file path
predictor = BankChurnPredictor(file_path=r"C:\Users\ALI\Desktop\Agents A\Bank Customer Churn Prediction.csv", 
                              output_dir=r"C:\Users\ALI\Desktop\Agents A\churn_model_outputs")

# Set matplotlib to not use GUI
import matplotlib
matplotlib.use('Agg')

# Run the full pipeline
try:
    best_model, best_threshold = predictor.run_full_pipeline()
    print(f"Best model: {best_model} with threshold {best_threshold:.4f}")
except Exception as e:
    import traceback
    print(f"An error occurred: {e}")
    traceback.print_exc()