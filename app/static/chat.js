// chat.js - Sends user messages to the Flask AI chat API
async function sendMessage() {
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const message = input.value.trim();
    if (!message) return;

    chatBox.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
    input.value = '';

    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    const data = await res.json();
    chatBox.innerHTML += `<div><strong>AI:</strong> ${data.response}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}

document.getElementById('user-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
});
