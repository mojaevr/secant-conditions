#!/usr/bin/env bash
# Создаёт git-репозиторий в этой папке и делает первый коммит.
# Использование: bash setup_git.sh
set -euo pipefail

cd "$(dirname "$0")"

# Удалить следы предыдущей попытки (если были)
if [ -d .git ]; then
  echo "==> .git уже существует. Удаляю и инициализирую заново."
  rm -rf .git
fi

# 1. Init
echo "==> git init -b main"
git init -b main

# 2. Имя/почта коммитера (локально для репо)
git config user.name  "Roman Mozhaev"
git config user.email "romamozh40@gmail.com"

# 3. Staging
echo "==> git add ."
git add .

# 4. Что попадёт в коммит
echo "==> Файлы в первом коммите:"
git ls-files | sed 's/^/   /'

# 5. Коммит
git commit -m "Initial commit: SP-Broyden master's thesis sources, notebooks, final PDF"

# 6. Подсказка для push
cat <<'EOF'

==============================================================
Локальный репозиторий готов. Теперь:

  1. Создай пустой public-репозиторий SP-Broyden на GitHub:
     https://github.com/new
     (имя: SP-Broyden, Public, без README/license/.gitignore)

  2. Добавь remote и запушь:
       git remote add origin git@github.com:<твой-username>/SP-Broyden.git
       git push -u origin main

     Через HTTPS (если не настроен SSH-ключ):
       git remote add origin https://github.com/<твой-username>/SP-Broyden.git
       git push -u origin main
     (потребуется Personal Access Token вместо пароля)

  3. Если установлен GitHub CLI, всё ещё проще:
       gh repo create SP-Broyden --public --source=. --remote=origin --push
==============================================================
EOF
