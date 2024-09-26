import streamlit as st
import joblib  # Assuming your models are saved as joblib files
import pickle

# Assuming you have functions to calculate SOi, Noi, Rpi, SPMi
def cal_SOi(so2):
    si=0
    if (so2<=40):
     si= so2*(50/40)
    elif (so2>40 and so2<=80):
     si= 50+(so2-40)*(50/40)
    elif (so2>80 and so2<=380):
     si= 100+(so2-80)*(100/300)
    elif (so2>380 and so2<=800):
     si= 200+(so2-380)*(100/420)
    elif (so2>800 and so2<=1600):
     si= 300+(so2-800)*(100/800)
    elif (so2>1600):
     si= 400+(so2-1600)*(100/800)
    return si

def cal_Noi(no2):
    ni=0
    if(no2<=40):
     ni= no2*50/40
    elif(no2>40 and no2<=80):
     ni= 50+(no2-40)*(50/40)
    elif(no2>80 and no2<=180):
     ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
     ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
     ni= 300+(no2-280)*(100/120)
    else:
     ni= 400+(no2-400)*(100/120)
    return ni

def cal_RSPMI(rspm):
    rpi=0
    if(rpi<=30):
     rpi=rpi*50/30
    elif(rpi>30 and rpi<=60):
     rpi=50+(rpi-30)*50/30
    elif(rpi>60 and rpi<=90):
     rpi=100+(rpi-60)*100/30
    elif(rpi>90 and rpi<=120):
     rpi=200+(rpi-90)*100/30
    elif(rpi>120 and rpi<=250):
     rpi=300+(rpi-120)*(100/130)
    else:
     rpi=400+(rpi-250)*(100/130)
    return rpi

def cal_SPMi(spm):
    spi=0
    if(spm<=50):
     spi=spm*50/50
    elif(spm>50 and spm<=100):
     spi=50+(spm-50)*(50/50)
    elif(spm>100 and spm<=250):
     spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
     spi=200+(spm-250)*(100/100)
    elif(spm>350 and spm<=430):
     spi=300+(spm-350)*(100/80)
    else:
     spi=400+(spm-430)*(100/430)
    return spi

# Load the selected model
def load_model():

    # Load Linear Regression model
    with open('Models_Final/linear_regression_model.pkl', 'rb') as f:
        linear_regression_model = pickle.load(f)

    # Load Decision Tree Regressor model
    with open('Models_Final/decision_tree_regressor_model.pkl', 'rb') as f:
        decision_tree_regressor_model = pickle.load(f)

    # Load Random Forest Regressor model
    with open('Models_Final/random_forest_regressor_model.pkl', 'rb') as f:
        random_forest_regressor_model = pickle.load(f)

    # Load Logistic Regression model
    with open('Models_Final/logistic_regression_model.pkl', 'rb') as f:
        logistic_regression_model = pickle.load(f)

    # Load Decision Tree Classifier model
    with open('Models_Final/decision_tree_classifier_model.pkl', 'rb') as f:
        decision_tree_classifier_model = pickle.load(f)

    # Load Random Forest Classifier model
    with open('Models_Final/random_forest_classifier_model.pkl', 'rb') as f:
        random_forest_classifier_model = pickle.load(f)

    # Load K-Nearest Neighbors model
    with open('Models_Final/knn_classifier_model.pkl', 'rb') as f:
        knn_classifier_model = pickle.load(f)

    return linear_regression_model, decision_tree_regressor_model, random_forest_regressor_model, logistic_regression_model, decision_tree_classifier_model, random_forest_classifier_model, knn_classifier_model

# Streamlit app
def main():
    
    linear_regression_model, decision_tree_regressor_model, random_forest_regressor_model, logistic_regression_model, decision_tree_classifier_model, random_forest_classifier_model, knn_classifier_model = load_model()
    
    st.title('Air Quality Prediction App')

    # Input fields for so2, no2, rspm, spm
    so2 = st.number_input('Enter SO2 value', min_value=0.0, step=0.1)
    no2 = st.number_input('Enter NO2 value', min_value=0.0, step=0.1)
    rspm = st.number_input('Enter RSPM value', min_value=0.0, step=0.1)
    spm = st.number_input('Enter SPM value', min_value=0.0, step=0.1)

    # Calculate SOi, Noi, Rpi, SPMi using your functions
    so_i = cal_SOi(so2)
    no_i = cal_Noi(no2)
    rp_i = cal_RSPMI(rspm)
    spm_i = cal_SPMi(spm)

    # Dropdown to select model
    model_name = st.selectbox("Select a model", ["Linear Regression Model", "Decision Tree Regressor Model", "Random Forest Regressor Model", "Logistic Regression Model", "Decision Tree Classifier Model", "Random Forest Classifier Model", "KNN Classifier Model"])  # Replace with your models

    # Predict button
    if st.button("Predict"):
        
        # If-else block to assign the selected model to the `model` variable
        if model_name == "Linear Regression Model":
            model = linear_regression_model
        elif model_name == "Decision Tree Regressor Model":
            model = decision_tree_regressor_model
        elif model_name == "Random Forest Regressor Model":
            model = random_forest_regressor_model
        elif model_name == "Logistic Regression Model":
            model = logistic_regression_model
        elif model_name == "Decision Tree Classifier Model":
            model = decision_tree_classifier_model
        elif model_name == "Random Forest Classifier Model":
            model = random_forest_classifier_model
        elif model_name == "KNN Classifier Model":
            model = knn_classifier_model
        else:
            st.error("Please select a valid model.")


        # Create input array for the model
        input_data = [[so_i, no_i, rp_i, spm_i]]
        
        
        # Perform prediction
        prediction = model.predict(input_data)
        air_quality = prediction[0]
        
        # Display the prediction result
        st.write(f"Prediction: {prediction[0]}")
        if model_name == "Decision Tree Classifier Model" or model_name == "Random Forest Classifier Model" or model_name == "KNN Classifier Model":
            if air_quality == 'Good':
                st.success(f"Air Quality: {air_quality}")
            elif air_quality == 'Moderate':
                st.info(f"Air Quality: {air_quality}")
            elif air_quality == 'Poor':
                st.warning(f"Air Quality: {air_quality}")
            elif air_quality in ['Unhealthy', 'Very Unhealthy']:
                st.error(f"Air Quality: {air_quality}")
            elif air_quality == 'Hazardous':
                st.error(f"Air Quality: {air_quality}", icon="🚨")
        elif model_name == "Linear Regression Model" or model_name == "Decision Tree Regressor Model":
            if air_quality>=0 and air_quality<=50:
                st.success(f"Air Quality: {air_quality} || Good")
            elif air_quality>=51 and air_quality<=150:
                st.info(f"Air Quality: {air_quality} || Moderate")
            elif air_quality>=151 and air_quality<=200:
                st.warning(f"Air Quality: {air_quality} || Unhealthy")
            elif air_quality>=201 and air_quality<=300:
                st.error(f"Air Quality: {air_quality} || Very Unhealthy")
            elif air_quality>=301:
                st.error(f"Air Quality: {air_quality} || Hazardous")
            else:
                st.write("Something is Wrong with the Inputs.")

if __name__ == '__main__':
    main()
