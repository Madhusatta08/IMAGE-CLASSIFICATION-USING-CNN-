import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy user store
users = {'admin': {'password': 'admin'},'madhu': {'password': 'madhu123'}}


class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None

# Load the trained model
model = tf.keras.models.load_model('cifar10_cnn.h5')

# Define classes for CIFAR-10
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format
    image = image.resize((32, 32))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Route to handle home page
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Route to handle login
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)

# Route to handle logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            processed_image = preprocess_image(filepath)
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = classes[predicted_class_index]
            return jsonify({'class': predicted_class_name})
    
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
