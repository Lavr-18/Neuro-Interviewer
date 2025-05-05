import logging
import os
import asyncio
from typing import Literal, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)
from dotenv import load_dotenv

load_dotenv()


openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в .env")
bot = Bot(token=TOKEN)
dp = Dispatcher(storage=MemoryStorage())


TEST_ROUNDS = 5



class TestStates(StatesGroup):
    choosing_position = State()
    confirming_readiness = State()
    in_test = State()
    answering = State()



position_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🧼 Горничная", callback_data="position_maid")],
    [InlineKeyboardButton(text="🛎 Администратор", callback_data="position_admin")],
    [InlineKeyboardButton(text="🔧 Техперсонал", callback_data="position_tech")]
])

ready_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="✅ Готов к тесту", callback_data="start_test")]
])


instruction_links = {
    "position_maid": [
        "Инструкция для горничной - часть 1: https://docs.google.com/document/...",
        "Инструкция для горничной - часть 2: https://docs.google.com/document/...",
        "Инструкция для горничной - часть 3: https://docs.google.com/document/..."
    ],
    "position_admin": ["Инструкция для администратора: https://docs.google.com/document/..."],
    "position_tech": ["Инструкция для техперсонала: https://docs.google.com/document/..."]
}


class TestData:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.results = []
        self.current_question = 0



@dp.message(Command("start"))
async def start(message: Message, state: FSMContext):
    await message.answer(
        "Добро пожаловать в систему тестирования сотрудников!\n"
        "Выберите вашу должность:",
        reply_markup=position_keyboard
    )
    await state.set_state(TestStates.choosing_position)
    await state.set_data({"test_data": TestData()})


@dp.callback_query(F.data.startswith("position_"))
async def choose_position(callback: CallbackQuery, state: FSMContext):
    position_key = callback.data
    await state.update_data(position=position_key)

    if position_key not in instruction_links:
        await callback.message.answer("Ошибка: выбранная должность не найдена")
        return

    instructions = instruction_links[position_key]
    instructions_text = "\n\n".join(instructions)

    await callback.message.answer(
        "Пожалуйста, внимательно ознакомьтесь с инструкциями для вашей должности:\n\n"
        f"{instructions_text}\n\n"
        "После изучения нажмите кнопку ниже, чтобы начать тестирование.",
        reply_markup=ready_keyboard
    )
    await state.set_state(TestStates.confirming_readiness)
    await callback.answer()



async def generate_question(instructions: list[str]) -> str:
    formatted_text = "\n\n".join(instructions)

    system_message = ChatCompletionSystemMessageParam(
        role="system",
        content=(
            "Ты — нейро-интервьюер для сотрудников отеля. "
            "Сгенерируй четкий вопрос, который проверит понимание "
            "одного из важных пунктов инструкции. Формулируй вопрос кратко и ясно."
        )
    )

    user_message = ChatCompletionUserMessageParam(
        role="user",
        content=f"Вот инструкция:\n\n{formatted_text}\n\nСгенерируй один вопрос по материалу."
    )

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",  # или "gpt-4" если доступно
        messages=[system_message, user_message],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message.content



async def evaluate_answer(question: str, answer: str, instructions: list[str]) -> tuple[bool, str]:
    formatted_text = "\n\n".join(instructions)

    system_message = ChatCompletionSystemMessageParam(
        role="system",
        content=(
            "Ты — нейро-интервьюер. Оцени ответ сотрудника на основе инструкции. "
            "Ответь в формате: 'CORRECT: да/нет, COMMENT: твой комментарий'"
        )
    )

    user_message = ChatCompletionUserMessageParam(
        role="user",
        content=(
            f"Инструкция:\n{formatted_text}\n\n"
            f"Вопрос: {question}\n"
            f"Ответ сотрудника: {answer}\n\n"
            "Оцени корректность ответа и дай краткий комментарий."
        )
    )

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
        temperature=0.3,
        max_tokens=200
    )

    result_text = response.choices[0].message.content
    is_correct = "CORRECT: да" in result_text.lower()
    comment = result_text.split("COMMENT:")[-1].strip()

    return is_correct, comment



@dp.callback_query(F.data == "start_test")
async def start_test(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    position_key = data.get("position")
    test_data: TestData = data.get("test_data")

    if position_key not in instruction_links:
        await callback.message.answer("Ошибка: должность не найдена.")
        return

    instructions = instruction_links[position_key]

    try:
        question = await generate_question(instructions)
        test_data.questions.append(question)
        await state.update_data(test_data=test_data)

        await callback.message.answer(f"Вопрос 1/{TEST_ROUNDS}:\n\n{question}")
        await state.set_state(TestStates.in_test)
    except Exception as e:
        logging.error(f"Ошибка при генерации вопроса: {e}")
        await callback.message.answer("Произошла ошибка при генерации вопроса. Попробуйте позже.")

    await callback.answer()



@dp.message(TestStates.in_test)
async def process_answer(message: Message, state: FSMContext):
    data = await state.get_data()
    position_key = data.get("position")
    test_data: TestData = data.get("test_data")

    if position_key not in instruction_links:
        await message.answer("Ошибка: должность не найдена.")
        return

    current_question_idx = len(test_data.answers)
    if current_question_idx >= len(test_data.questions):
        await message.answer("Ошибка: вопрос не найден.")
        return

    question = test_data.questions[current_question_idx]
    user_answer = message.text
    test_data.answers.append(user_answer)

    # Оценка ответа
    instructions = instruction_links[position_key]
    is_correct, comment = await evaluate_answer(question, user_answer, instructions)
    test_data.results.append(is_correct)

    await message.answer(
        f"Результат:\n\n"
        f"{'✅ Правильно' if is_correct else '❌ Неправильно'}\n"
        f"Комментарий: {comment}"
    )

    # Проверяем, завершен ли тест
    if len(test_data.answers) >= TEST_ROUNDS:
        await finish_test(message, state, test_data)
    else:
        # Генерируем следующий вопрос
        try:
            next_question = await generate_question(instructions)
            test_data.questions.append(next_question)
            await state.update_data(test_data=test_data)

            await message.answer(
                f"Вопрос {len(test_data.answers) + 1}/{TEST_ROUNDS}:\n\n"
                f"{next_question}"
            )
        except Exception as e:
            logging.error(f"Ошибка при генерации вопроса: {e}")
            await message.answer("Произошла ошибка. Тест прерван.")
            await state.clear()


# Завершение теста
async def finish_test(message: Message, state: FSMContext, test_data: TestData):
    correct_answers = sum(test_data.results)
    score = correct_answers / TEST_ROUNDS

    if score >= 0.8:
        result = "✅ Пройдено"
    elif score >= 0.5:
        result = "⚠️ Требуется дообучение"
    else:
        result = "❌ Не пройдено"

    # Формируем отчет
    report = (
        f"Тестирование завершено!\n\n"
        f"Результат: {result}\n"
        f"Правильных ответов: {correct_answers}/{TEST_ROUNDS}\n\n"
        f"Детали:\n"
    )

    for i, (question, answer, is_correct) in enumerate(zip(
            test_data.questions, test_data.answers, test_data.results
    ), 1):
        report += (
            f"\n{i}. {'✅' if is_correct else '❌'}\n"
            f"Вопрос: {question}\n"
            f"Ответ: {answer}\n"
        )

    await message.answer(report)

    # Здесь должна быть логика сохранения результатов гугл таблицу
    # save_results_to_db(message.from_user.id, test_data)

    await state.clear()


# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Запуск бота
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())