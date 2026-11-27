# Общий репозиторий команды

Адрес: https://github.com/Lysyyyyy/project1

Участники: Масан Влад и Медведев Федор, 4 курс ИБ. Владелец репозитория добавляет аккаунт второго участника через Settings → Collaborators; второй участник принимает приглашение. Наличие такого доступа проверяется самими участниками.

## Рабочая копия каждого участника

```powershell
git clone https://github.com/Lysyyyyy/project1.git
cd project1
git switch -c work/my-task
```

Каждый задаёт своё имя и настоящий email для новых коммитов через `git config user.name` и `git config user.email`. Вносите изменение, запускайте проверки, выполняйте `git add` для изменённых файлов, `git commit` и `git push -u origin work/my-task`. На GitHub создайте Pull Request. Второй участник проверяет его перед объединением. Для новой задачи используйте отдельное имя ветки.

Чтобы обновить main:

```powershell
git switch main
git pull --ff-only
python -m unittest -v test_system test_extended test_assistant
```

В папке Материалы есть общий ZIP с автономным локальным bare-репозиторием. Он относится к варианту передачи комплекта без GitHub. Для обычной работы используйте удалённый origin, заданный командой clone выше.

Исходная история подготовлена от технического автора Project scaffold. Личная хронология работы студентов из неё не восстанавливается. Фактическое распределение ответственности подтверждено в TASKS.md и docs/команда.md.
