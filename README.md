Docker (сборка и запуск с .env)

- Сборка образа:
  docker build -t langpipe:latest .

- Запуск с переменными окружения из .env:
  docker run --rm -p 8000:8000 --env-file .env langpipe:latest

- Проверка:
  Откройте http://localhost:8000 или выполните: curl http://localhost:8000/config

