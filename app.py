from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    sentence = data.get('sentence')
    return jsonify({"sentence": sentence})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
