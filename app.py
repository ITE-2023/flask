from flask import Flask

app = Flask(__name__)

@app.route('/sentiment')
def sentiment():
    return {"노래 제목":"제목", "가수":"가수", "이미지": "URL"}




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)