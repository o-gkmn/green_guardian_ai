import os
from flask import Flask, request, jsonify
from MLAlgorithm import MLAlgorithm

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        file_path = f'data/real_time_data/{audio_file.filename}'
        filename = audio_file.filename

        audio_file.save(file_path)
        ml = MLAlgorithm(filePath= file_path)
        prediction = ml.make_predict()
        print(prediction)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return jsonify({'error': f'File {filename} not found'}), 404

        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



app.run(debug=True)