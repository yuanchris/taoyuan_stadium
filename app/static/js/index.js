
// control server
async function start() {
    const show = document.querySelector('#show_server');
    const message = await fetch('start_control', {
        method: 'GET',
    }).then((res => res.json()))  
    console.log(message);
    if (message.error){
        show.innerHTML = 'start: ' + message.error;
    } else {
        console.log('started server ');
        show.innerHTML = 'started server ';
    }
}
async function stop() {
    const show = document.querySelector('#show_server');
    const message = await fetch('stop_control', {
        method: 'GET',
    }).then((res => res.json()))  
    console.log(message);
    if (message.error){
        show.innerHTML = 'stop: ' + message.error;
    } else {
        console.log('stopped server');
        show.innerHTML = 'stopped server';
    }
}
// control redis
async function start_redis() {
    const show = document.querySelector('#show_redis');
    const message = await fetch('start_redis', {
        method: 'GET',
    }).then((res => res.json()))  
    console.log(message);
    if (message.error){
        show.innerHTML = 'start: ' + message.error;
    } else {
        console.log('started redis ');
        show.innerHTML = 'started redis ';
    }
}
async function stop_redis() {
    const show = document.querySelector('#show_redis');
    const message = await fetch('stop_redis', {
        method: 'GET',
    }).then((res => res.json()))  
    console.log(message);
    if (message.error){
        show.innerHTML = 'stop: ' + message.error;
    } else {
        console.log('stopped redis');
        show.innerHTML = 'stopped redis';
    }
}

async function get_player_list() {
    const defense = document.querySelector('#defense');
    const attack = document.querySelector('#attack');
    const message = await fetch('get_player_list', {
        method: 'GET',
    }).then((res => res.json()))  
    console.log(message);

    if (message.error){
        console.log(message.error)
    } else {
        attack.innerHTML = '';
        const attacker = message.datas.attack;
        for(let i = 0; i < attacker.length; i++) {
            attack.innerHTML += `<p>第${i+1}棒: ${attacker[i]}`;
        }

        defense.innerHTML = ''
        const defender = message.datas.defend;
        for(item in defender) {
            defense.innerHTML += `<p>${item}: ${defender[item]}`;
        }
    }
}

async function track_board(){
    // window.open("track_board.html", "trackboard", "height=700,width=1200,\
    //     toolbar=yes,menubar=yes,scrollbars=yes, resizable=yes,location=yes, status=yes");
    window.open("track_board.html", "height=700,width=1200,\
        toolbar=yes,menubar=yes,scrollbars=yes,location=yes,status=yes,resizable=yes");

}