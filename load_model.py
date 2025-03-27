import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model from file
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if model_data is a dictionary containing model and preprocessors
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data
        else:
            # If it's just the model
            return {'model': model_data, 'preprocessors': None}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Streamlit UI for predictions
def predict():
    st.title("Model Prediction App")
    
    model_data = load_model()
    if not model_data:
        st.stop()
    
    model = model_data['model']
    preprocessors = model_data.get('preprocessors')
    
    # Display information about the model
    st.write("## Model Information")
    model_type = type(model).__name__
    st.write(f"Model type: {model_type}")
    
    # Create input fields for each feature
    st.write("## Enter Feature Values")
    Brand = st.number_input('Enter value for Brand', value=0.0, step=0.1)
    Processor_Speed = st.number_input('Enter value for Processor_Speed', value=0.0, step=0.1)
    RAM_Size = st.number_input('Enter value for RAM_Size', value=0.0, step=0.1)
    Storage_Capacity = st.number_input('Enter value for Storage_Capacity', value=0.0, step=0.1)
    Screen_Size = st.number_input('Enter value for Screen_Size', value=0.0, step=0.1)
    Weight = st.number_input('Enter value for Weight', value=0.0, step=0.1)
    
    # Predict the output
    if st.button("Predict"):
        try:
            # Create a DataFrame with the input values
            input_data = {
                'Brand': Brand, 'Processor_Speed': Processor_Speed, 'RAM_Size': RAM_Size, 'Storage_Capacity': Storage_Capacity, 'Screen_Size': Screen_Size, 'Weight': Weight
            }
            input_df = pd.DataFrame([input_data])
            
            # Apply any preprocessing if available
            if preprocessors:
                # Handle categorical features if encoder is available
                if 'categorical_encoder' in preprocessors and preprocessors['categorical_encoder']:
                    categorical_encoder = preprocessors['categorical_encoder']
                    categorical_features = preprocessors.get('categorical_features', [])
                    
                    if categorical_features and categorical_encoder:
                        # Apply one-hot encoding or label encoding
                        try:
                            # For one-hot encoding
                            encoded_cats = categorical_encoder.transform(input_df[categorical_features])
                            if hasattr(encoded_cats, 'toarray'):  # For sparse matrices
                                encoded_cats = encoded_cats.toarray()
                            
                            # Get the feature names after transformation
                            if hasattr(categorical_encoder, 'get_feature_names_out'):
                                cat_cols = categorical_encoder.get_feature_names_out(categorical_features)
                                encoded_df = pd.DataFrame(encoded_cats, columns=cat_cols)
                                
                                # Drop original categorical columns and add encoded ones
                                input_df = input_df.drop(columns=categorical_features)
                                input_df = pd.concat([input_df, encoded_df], axis=1)
                        except Exception as e:
                            st.warning(f"Error applying categorical encoding: {e}")
                
                # Handle numerical scaling if available
                if 'scaler' in preprocessors and preprocessors['scaler']:
                    scaler = preprocessors['scaler']
                    numerical_features = preprocessors.get('numerical_features', [])
                    
                    if numerical_features and scaler:
                        try:
                            # Scale numerical features
                            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
                        except Exception as e:
                            st.warning(f"Error applying scaling: {e}")
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Display the prediction
            st.write("## Prediction Result")
            
            # Check if it's a classification or regression model
            if hasattr(model, 'classes_'):
                # Classification model
                st.write(f"Predicted class: {prediction[0]}")
                
                # If model has predict_proba method, show probabilities
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(input_df)
                        st.write("### Class Probabilities")
                        for i, class_name in enumerate(model.classes_):
                            st.write(f"{class_name}: {proba[0][i]:.4f}")
                    except:
                        pass
            else:
                # Regression model
                st.write(f"Predicted value: {prediction[0]:.4f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error(f"Details: {str(e)}")

if __name__ == "__main__":
    predict()