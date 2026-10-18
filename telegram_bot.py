import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    ContextTypes,
    CommandHandler,
    filters,
)

from agent import agent

load_dotenv()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm your AI assistant. Ask me anything."
    )


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": update.message.text,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(update.effective_chat.id)
            }
        },
    )

    await update.message.reply_text(result["messages"][-1].content)


app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("Bot is running...")
app.run_polling()