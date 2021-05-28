from flask import Flask, render_template, request
import socket, json
from flask_socketio import SocketIO,emit
import time

app = Flask(__name__)
socketio = SocketIO(app)
# === socket ===
@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)

@socketio.on('tracking')
def tracking(message):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8062))
        sock.settimeout(1)
        # sock.setblocking(False)
        receive_message = sock.recv(81920).decode('utf-8')
        result = json.loads(receive_message)
        emit('send_track', result)
    except BlockingIOError as e:
        # print("BlockingIOError: ", e)
        pass    
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})
    
    # print('===receive_message====',receive_message)
    
    sock.close()
    # return json.dumps(receive_message)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/track_board.html")
def view():
    return render_template("track_board.html")

@app.route("/start_control")
def start_control():
    print("=== start control ===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8050))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})

    message = '{"message":"start"}'
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)


@app.route("/stop_control")
def stop_control():
    print("===stop capture===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8050))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})

    message = '{"message":"stop"}'
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)


@app.route("/get_player_list")
def get_player_list():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8051))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})

    message = '{"message":"player_list"}'
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)


@app.route("/start_redis")
def start_redis():
    print("=== start control ===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8051))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})

    message = '{"message":"redis_start"}'
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)


@app.route("/stop_redis")
def stop_redis():
    print("===stop capture===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8051))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})

    message = '{"message":"redis_stop"}'
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)

@app.route("/hugo_game_information", methods=['GET', 'POST'])
def hugo_game_information():
    print("===hugo_game_information===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print("Error:", e)
        return json.dumps({"error": "establish error"})

    try:
        sock.connect(("127.0.0.1", 8051))
    except socket.error as e:
        print("[ERROR] ", e)
        return json.dumps({"error": "connect error"})
    req =  request.values.to_dict()
    print('req', req)
    message = {"message":"hugo_game_information", "data":req}
    message = json.dumps(message)
    sock.send(message.encode("UTF-8"))
    receive_message = sock.recv(1024).decode("utf-8")
    receive_dict = json.loads(receive_message)
    sock.close()
    return json.dumps(receive_dict)



if __name__ == "__main__":
    app.debug = True
    # app.run()
    socketio.run(app, host="127.0.0.1", port=5000)