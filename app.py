#app.py
from flask import Flask, render_template, send_file, session, request, jsonify
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' # Required for sessions

# --- ROUTES FOR PAGES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signin')
def signin_page():
    return render_template('signin.html')

# Route for the Chemistry page inside the subfolder
@app.route('/chemistry-course')
def chemistry_course():
    return render_template('Chemistry/AI_for_chem.html')

# --- AUTHENTICATION API (Talking to your HTML) ---

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # 1. VERIFY USER (Mock logic for now - connect to DB later)
    # In a real app, you would query your database here.
    # For now, we simulate success if fields are filled.
    if email and password:
        session['user'] = email # Set Python Session
        return jsonify({"status": "success", "message": "Logged in via Python"})
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 400

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    # In a real app, save 'data' to database here
    session['user'] = data.get('email')
    return jsonify({"status": "success", "message": "Registered via Python"})

@app.route('/api/logout')
def api_logout():
    session.clear()
    return jsonify({"status": "success"})

# --- FILE SERVING (The Integration) ---

@app.route('/api/file/<filename>')
def serve_file(filename):
    # SECURITY CHECK: Is the user logged in to Python?
    if 'user' not in session:
        # If not logged in, return 401 Unauthorized
        return "You must be logged in to download this file.", 401
    
    try:
        # Path to your PDFs in static/chemistry_pdfs
        file_path = os.path.join('static/chemistry_pdfs', filename)
        return send_file(file_path)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)