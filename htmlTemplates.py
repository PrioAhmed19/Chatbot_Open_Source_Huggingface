# ---------- CSS ----------
css = '''
<style>
/* Chat container */
.chat-container {
    max-width: 800px;
    margin: auto;
    font-family: "Segoe UI", Roboto, sans-serif;
}

/* Shared message style */
.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    transition: all 0.2s ease-in-out;
}
.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}

/* User and Bot background colors */
.chat-message.user {
    background: linear-gradient(135deg, #2b313e, #3a4250);
}
.chat-message.bot {
    background: linear-gradient(135deg, #475063, #5a6475);
}

/* Avatar */
.chat-message .avatar {
    flex-shrink: 0;
}
.chat-message .avatar img {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 2px solid #fff;
    object-fit: cover;
    box-shadow: 0 2px 4px rgba(0,0,0,0.25);
}

/* Message bubble */
.chat-message .message {
    flex-grow: 1;
    padding: 0.6rem 1rem;
    border-radius: 0.75rem;
    background-color: rgba(255,255,255,0.05);
    color: #f5f5f5;
    line-height: 1.5;
    font-size: 0.95rem;
    word-wrap: break-word;
}
</style>
'''

# ---------- Templates ----------
bot_template = '''
<div class="chat-container">
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://www.pngwing.com/en/free-png-zvjoi">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
</div>
'''

user_template = '''
<div class="chat-container">
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://www.pngwing.com/en/free-png-yiazz">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
</div>
'''
