from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_key_change_in_production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/image', exist_ok=True)

# Attack type mapping
ATTACK_TYPES = {
    0: "DoS Attack",
    1: "Probe Attack", 
    2: "R2L Attack",
    3: "U2R Attack",
    4: "Normal Traffic"
}

# Build and load model
def build_model():
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv1D(64, 3, padding="same", input_shape=(25, 1), 
                     activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Conv1D(64, 3, padding="same", activation='relu', 
                     kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 3, padding="same", activation='relu', 
                     kernel_regularizer=l2(0.001)))
    model.add(Conv1D(128, 3, padding="same", activation='relu', 
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling1D(pool_size=2))
    
    # Batch Normalization
    model.add(BatchNormalization())
    
    # LSTM layer
    model.add(LSTM(units=100, return_sequences=False, dropout=0.1, 
                   kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(units=5, activation='softmax', kernel_regularizer=l2(0.01)))
    
    # Compile
    opt = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

# Load model
model = build_model()
try:
    model.load_weights('cnnlstm.h5')
    print("Model loaded successfully!")
except:
    print("Warning: Model weights not found. Please train the model first.")

# Load selected features
def load_selected_features():
    try:
        dataset = pd.read_csv('selectedfeature.csv')
        columns = dataset.columns
        first_row = dataset.iloc[0].values.tolist()
        selected_columns = []
        for i in range(len(first_row)):
            if first_row[i] == 1.0:
                selected_columns.append(columns[i])
        return selected_columns
    except Exception as e:
        print(f"Error loading selected features: {e}")
        return None

selected_features = load_selected_features()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_data(df, selected_cols):
    """Prepare data for model prediction"""
    try:
        # Select only the required columns
        data = df[selected_cols].values
        # Reshape for CNN-LSTM input
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data
    except Exception as e:
        raise ValueError(f"Error preparing data: {e}")

def predict_traffic(data):
    """Predict network traffic type"""
    predictions = model.predict(data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    return predicted_classes, confidences

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if selected_features is None:
            return jsonify({'error': 'Selected features not loaded. Please check selectedfeature.csv'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process CSV
        df = pd.read_csv(filepath)
        
        # Prepare data
        X_test = prepare_data(df, selected_features)
        
        # Make predictions
        predictions, confidences = predict_traffic(X_test)
        
        # Prepare results
        results = []
        attack_count = {key: 0 for key in ATTACK_TYPES.keys()}
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            attack_type = ATTACK_TYPES[int(pred)]
            attack_count[int(pred)] += 1
            is_attack = int(pred) != 4

            
            results.append({
                'index': int(i + 1),
                'prediction': attack_type,
                'confidence': f"{float(conf) * 100:.2f}%",
                'is_attack': bool(int(pred) != 4)
            })
        
        # Calculate statistics
        total = int(len(predictions))
        normal_count = int(attack_count[4])
        attack_total = int(total - normal_count)
        
        stats = {
            'total_records': total,
            'normal_traffic': normal_count,
            'total_attacks': attack_total,
            'attack_percentage': f"{(attack_total/total)*100:.2f}%",
            'attack_breakdown': {
                ATTACK_TYPES[k]: int(v) for k, v in attack_count.items() if k != 4
            }
        }
        
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'results': results[:100],  # Limit to first 100 for display
            'statistics': stats,
            'total_predictions': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)