
from flask import Flask
from factories import CnnFactory

app = Flask(__name__)



@app.route('/predict', methods=['GET'])
def predict2():
    detection = CnnFactory.create()
    response = detection.first_rule(True)
    return {
        "success": "success"
    }


if __name__ == "__main__":
    PORT = 5000
    app.run(port=PORT)