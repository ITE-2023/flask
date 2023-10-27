from flask import Flask, request, jsonify
from model.sentiment_call import recommend

app = Flask(__name__)


@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    content = data.get('content')
    emotion, musicList = recommend(content)
    return jsonify(
        {
            "sentiment": emotion,
            "musicList": musicList
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
