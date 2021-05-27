var image = document.getElementById('image');
// ======  websocket version =====
// let ws = new WebSocket('ws://127.0.0.1:8062')
// ws.binaryType = 'arraybuffer';
// ws.onmessage = function (event) {
//     var arrayBuffer = event.data;
//     image.src = "data:image/jpeg;base64," + encode(new Uint8Array(arrayBuffer));
// };
// window.onbeforeunload = function(){
//     console.log('test')
//     ws.onclose = function() {}; // disable onclose handler first 
//     ws.close() 
//  }
var socket = io();
// socket.on('connect', function() {
//     socket.emit('my event', {data: 'I\'m connected!'});
// });

socket.on('connect', function() {
    setInterval(function(){
        socket.emit('tracking', {data: 'tacked board website connected!'});
        // console.log('send tracking');
    }, 1000);
    // setTimeout(function(){
    //     socket.emit('tracking', {data: 'tacked board website connected!'});
    //     // console.log('send tracking');
    // }, 1000);
    // console.log('send tracking')
});




socket.on('send_track', function(msg) {
    console.log(msg);
    let canvas = document.getElementById("canvas");
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let circle_x, circle_y;
    for (let key  in msg) {
        circle_x = msg[key][0]
        circle_y = msg[key][1]
        ctx.beginPath();
        //寬度及色彩設定
        ctx.lineWidth = 1;
        if (key == 'H') {
            ctx.strokeStyle = "#FF0000"
            ctx.fillStyle = "#FF0000"
        } else {
            ctx.strokeStyle = "#0000FF"
            ctx.fillStyle = "#0000FF"
        }

        /*使用arc(x,y,r,s,e)畫一個圓
        x,y是圓心的座標，r是半徑，s和e是起點和終點的角度*/
        ctx.arc(circle_x,circle_y,2,0,Math.PI*2)
        ctx.fill()
        ctx.stroke()

        //text
        ctx.font="8px serif";
        ctx.fillText(key + ' ' + msg[key][2],circle_x + 4,circle_y+2);

    }

 

    // let arrayBuffer = msg;
    // if (arrayBuffer.length != 0) {
    //     image.src = "data:image/jpeg;base64," + encode(new Uint8Array(arrayBuffer));
    // }
});

// socket.on('disconnect', function() {
//     socket.emit('my event', {data: 'I\'m not connected!'});
// });

function encode (input) {
    var keyStr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
    var output = "";
    var chr1, chr2, chr3, enc1, enc2, enc3, enc4;
    var i = 0;

    while (i < input.length) {
        chr1 = input[i++];
        chr2 = i < input.length ? input[i++] : Number.NaN; // Not sure if the index
        chr3 = i < input.length ? input[i++] : Number.NaN; // checks are needed here

        enc1 = chr1 >> 2;
        enc2 = ((chr1 & 3) << 4) | (chr2 >> 4);
        enc3 = ((chr2 & 15) << 2) | (chr3 >> 6);
        enc4 = chr3 & 63;

        if (isNaN(chr2)) {
            enc3 = enc4 = 64;
        } else if (isNaN(chr3)) {
            enc4 = 64;
        }
        output += keyStr.charAt(enc1) + keyStr.charAt(enc2) +
                  keyStr.charAt(enc3) + keyStr.charAt(enc4);
    }
    return output;
}