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
    let mysvg = document.getElementById('mysvg')

    // $("#mysvg").empty();
    mysvg.innerHTML = '<image width="417" height="358" xlink:href="/static/img/field.svg"/>'
    // var circle= makeSVG('circle', {cx: 100, cy: 50, r:40, stroke: 'black', 'stroke-width': 2, fill: 'red'});
    if (typeof msg['pause_start'] !== 'undefined') {
        let text= makeText('text', {x: '50%', y: '40%', fill: 'red', style:"font-size: 28px" ,"dominant-baseline":"middle", "text-anchor":"middle"}, 'Pausing');
        mysvg.appendChild(text);
    }
    else {
        for (let key  in msg) {
            let circle_x = msg[key][0]
            let circle_y = msg[key][1]
            let circle = makeCircle('circle', {cx: circle_x, cy: circle_y, r:5, fill: '#1E90FF'});
            if (key == 'H') {
                circle= makeCircle('circle', {cx: circle_x, cy: circle_y, r:5, fill: '#FF4500'});
            }
            let location = makeText('text', {x: circle_x, y: circle_y, fill: 'white', style:"font-size: 6px", "dominant-baseline":"middle", "text-anchor":"middle"}, key);
            let text= makeText('text', {x: circle_x + 7, y: circle_y + 2, fill: 'black', style:"font-size: 8px"}, msg[key][2]);
            
            mysvg.appendChild(circle);
            mysvg.appendChild(text);
            mysvg.appendChild(location);
    
    
            //text
            // ctx.font="8px serif";
            // ctx.fillText(key + ' ' + msg[key][2],circle_x + 4,circle_y+2);
    
        }

    }

});

// socket.on('disconnect', function() {
//     socket.emit('my event', {data: 'I\'m not connected!'});
// });

function makeCircle(tag, attrs) {
    var el= document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs)
        el.setAttribute(k, attrs[k]);
    return el;
}
function makeText(tag, attrs, text) {
    var el= document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs)
        el.setAttribute(k, attrs[k]);
    el.innerHTML = text;
    return el;
}


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