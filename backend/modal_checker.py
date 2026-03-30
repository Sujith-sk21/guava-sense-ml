"""
Script to check and verify your existing .pkl model file
Run this to see if your model is compatible with the backend
"""

import pickle
import sys


def check_model_file(pkl_path):
    """Check what's inside your .pkl file"""

    print("=" * 60)
    print("🔍 CHECKING MODEL FILE")
    print("=" * 60)
    print(f"File: {pkl_path}\n")

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        print("✅ File loaded successfully!\n")

        # Check if it's a dictionary (the format backend expects)
        if isinstance(data, dict):
            print("📦 Model Format: Dictionary (Good!)")
            print(f"\nKeys in the file: {list(data.keys())}")

            # Check for required keys
            required_keys = ['model', 'selected_features', 'label_map']
            missing_keys = [key for key in required_keys if key not in data]

            if not missing_keys:
                print("\n✅ All required keys present!")
                print(f"   - Model: {type(data['model'])}")
                print(f"   - Selected features: {len(data['selected_features'])} features")
                print(f"   - Label map: {data['label_map']}")
                print(f"   - Classes: {list(data['label_map'].values())}")

                print("\n🎉 Your model file is COMPATIBLE with the backend!")
                print("   Just place it in the 'models/' folder as 'guava_disease_model.pkl'")
                return True
            else:
                print(f"\n⚠️ Missing required keys: {missing_keys}")
                print("\n📝 Your file needs to be converted. See instructions below.")
                return False

        else:
            print(f"📦 Model Format: {type(data)}")
            print("\n⚠️ Model is not in dictionary format.")

            # Try to detect what it is
            if hasattr(data, 'predict'):
                print("   Detected: Direct sklearn model object")
                print("\n📝 Your file needs to be converted. See instructions below.")

            return False

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


def create_compatible_model(original_pkl, output_path='guava_disease_model.pkl'):
    """
    If your model needs conversion, use this function
    You'll need to provide the missing information manually
    """

    print("\n" + "=" * 60)
    print("🔧 MODEL CONVERSION HELPER")
    print("=" * 60)

    try:
        with open(original_pkl, 'rb') as f:
            original_data = pickle.load(f)

        # If it's just the model
        if hasattr(original_data, 'predict') and not isinstance(original_data, dict):
            print("\n⚠️ Your pickle file contains only the model.")
            print("   You need to provide additional information:")
            print("\n   1. selected_features: List of feature indices used")
            print("   2. label_map: Dictionary mapping class indices to names")
            print("\n   Example:")
            print("   selected_features = [0, 5, 12, 18, ...]  # indices from 0-167")
            print("   label_map = {0: 'Healthy', 1: 'Disease1', 2: 'Disease2'}")
            print("\n📝 To convert, manually create a new file:")
            print("""
import pickle

# Your original model
with open('your_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create compatible package
model_package = {
    'model': model,
    'selected_features': [0, 1, 2, ...],  # ADD YOUR FEATURE INDICES HERE
    'label_map': {0: 'Class1', 1: 'Class2', ...}  # ADD YOUR CLASSES HERE
}

# Save compatible version
with open('guava_disease_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Compatible model created!")
            """)
            return False

        elif isinstance(original_data, dict):
            print("✅ Your model is already in the correct format!")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <path_to_your_model.pkl>")
        print("\nExample: python check_model.py guava_model.pkl")
        sys.exit(1)

    pkl_path = sys.argv[1]

    # Check the model
    is_compatible = check_model_file(pkl_path)

    if not is_compatible:
        print("\n" + "=" * 60)
        print("💡 WHAT TO DO NEXT")
        print("=" * 60)
        print("\nOption 1: Run the Model Saver script in your Colab notebook")
        print("   - This will create a properly formatted .pkl file")
        print("   - Use the 'Model Saver Script' provided earlier")

        print("\nOption 2: Look for additional variables in your Colab")
        print("   - Check if you saved 'selected_features' and 'label_map' separately")
        print("   - These might be in separate .pkl files or variables")

        print("\nOption 3: If you have the training code, find these values:")
        print("   - selected_features: Output from genetic algorithm")
        print("   - label_map: Dictionary of class names (from your dataset)")
        print("   - Then manually create the compatible format")

        create_compatible_model(pkl_path)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()