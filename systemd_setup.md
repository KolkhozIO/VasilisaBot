# Настройка SummaryBot как systemd сервиса

Эта инструкция поможет вам настроить запуск двух языковых версий SummaryBot (русской и английской) как systemd сервисов.

## Подготовка сервисных файлов

1. Отредактируйте файлы `summarybot-en.service` и `summarybot-ru.service`, заменив:
   - `user` на имя вашего пользователя
   - `/home/user/` на путь к директории с ботом
   - `YOUR_EN_BOT_TOKEN` и `YOUR_RU_BOT_TOKEN` на токены ваших ботов

## Установка сервисов

1. Скопируйте сервисные файлы в системную директорию:

```bash
sudo cp summarybot-en.service /etc/systemd/system/
sudo cp summarybot-ru.service /etc/systemd/system/
```

2. Перезагрузите конфигурацию systemd:

```bash
sudo systemctl daemon-reload
```

3. Включите автозапуск сервисов при загрузке системы:

```bash
sudo systemctl enable summarybot-en.service
sudo systemctl enable summarybot-ru.service
```

## Управление сервисами

### Запуск сервисов:

```bash
sudo systemctl start summarybot-en.service
sudo systemctl start summarybot-ru.service
```

### Проверка статуса:

```bash
sudo systemctl status summarybot-en.service
sudo systemctl status summarybot-ru.service
```

### Остановка сервисов:

```bash
sudo systemctl stop summarybot-en.service
sudo systemctl stop summarybot-ru.service
```

### Перезапуск сервисов:

```bash
sudo systemctl restart summarybot-en.service
sudo systemctl restart summarybot-ru.service
```

## Просмотр логов

Для просмотра логов используйте команду:

```bash
journalctl -u summarybot-en.service -f
journalctl -u summarybot-ru.service -f
```

Флаг `-f` позволяет следить за логами в реальном времени.

## Важные замечания

1. Каждый бот будет использовать свою собственную директорию для данных, основанную на последних 8 символах токена бота.
2. Данные будут храниться в поддиректориях `/home/user/data/[последние 8 символов токена]/`.
3. Убедитесь, что у пользователя, от имени которого запускается сервис, есть права на запись в директорию с данными.