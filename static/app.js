let ws;
let audioContext;
let processor;
let mediaStream;
let isRecording = false;
let pulseTimeout;
let nextPlaybackTime = 0;
let activeSources = [];

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const hal = document.querySelector('.animation');
hal.classList.add('idle');

startBtn.addEventListener('click', async () => {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = handleMessage;
    await startAudio();
    startBtn.disabled = true;
    stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
    stopAudio();
    ws.close();
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

function handleMessage(event) {
    const data = JSON.parse(event.data);
    if (data.event === 'clear') {
        clearAudio();
        return;
    }
    if (data.audio) {
        const binary = atob(data.audio);
        const buf = new ArrayBuffer(binary.length);
        const view = new Uint8Array(buf);
        for (let i = 0; i < binary.length; i++) {
            view[i] = binary.charCodeAt(i);
        }
        const int16 = new Int16Array(buf);
        const pcm = int16ToPCM(int16);
        playAudio(pcm);
    }
}

async function startAudio() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
    nextPlaybackTime = audioContext.currentTime;
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(mediaStream);
    if (audioContext.audioWorklet) {
        await audioContext.audioWorklet.addModule('/worklet.js');
        processor = new AudioWorkletNode(audioContext, 'capture-processor');
        processor.port.onmessage = e => {
            const pcm16 = e.data;
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(pcm16.buffer);
            }
        };
    } else {
        processor = audioContext.createScriptProcessor(1024, 1, 1);
        processor.onaudioprocess = e => {
            const input = e.inputBuffer.getChannelData(0);
            const pcm16 = floatTo16BitPCM(input);
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(pcm16.buffer);
            }
        };
    }
    source.connect(processor);
    processor.connect(audioContext.destination);
    isRecording = true;
}

function stopAudio() {
    if (processor) {
        processor.disconnect();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    nextPlaybackTime = 0;
    isRecording = false;
    activeSources = [];
    hal.classList.remove('speaking');
    hal.classList.add('idle');
}

function clearAudio() {
    activeSources.forEach(src => {
        try { src.stop(); } catch (e) {}
    });
    activeSources = [];
    if (audioContext) {
        nextPlaybackTime = audioContext.currentTime;
    } else {
        nextPlaybackTime = 0;
    }
    hal.classList.remove('speaking');
    hal.classList.add('idle');
}

function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
}


function int16ToPCM(int16) {
    const pcm = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        pcm[i] = int16[i] / 32768;
    }
    return pcm;
}

function playAudio(pcm) {
    const buffer = audioContext.createBuffer(1, pcm.length, 24000);
    buffer.copyToChannel(pcm, 0);
    const src = audioContext.createBufferSource();
    src.buffer = buffer;
    src.connect(audioContext.destination);
    if (nextPlaybackTime < audioContext.currentTime) {
        nextPlaybackTime = audioContext.currentTime;
    }
    src.start(nextPlaybackTime);
    activeSources.push(src);
    hal.classList.remove('idle');
    hal.classList.add('speaking');
    src.onended = () => {
        activeSources = activeSources.filter(s => s !== src);
        if (activeSources.length === 0) {
            hal.classList.remove('speaking');
            hal.classList.add('idle');
        }
    };
    nextPlaybackTime += buffer.duration;
}
