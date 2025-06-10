# Django REST API Server - インストールガイド

> 📋 **venv環境でのDjango REST APIサーバー完全セットアップガイド**

## 目次
- [前提条件](#前提条件)
- [プロジェクトセットアップ](#プロジェクトセットアップ)
- [環境設定](#環境設定)
- [データベースセットアップ](#データベースセットアップ)
- [開発サーバー](#開発サーバー)
- [VS Code設定](#vs-code設定)
- [インストール確認](#インストール確認)
- [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

### 必要なソフトウェア
- **Python 3.11+** (推奨)
- **PostgreSQL 12+** (本番データベース用)
  - ⚠️ **PostGIS不要**: 位置情報検索はPostgreSQLの標準数学関数で実装済み
- **Redis 6+** (キャッシュとJWT検証用)
- **Git** (バージョン管理用)
- **VS Code** (推奨IDE)

### システム依存関係 (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip postgresql postgresql-contrib redis-server git

# Redis サーバー起動
sudo service redis-server start

# Redis 接続確認
redis-cli ping  # PONG が返ることを確認
```

### Docker環境での前提条件
- **認証サーバー**: `host.docker.internal:8080` で稼働
- **PostgreSQL**: ローカルまたは Docker で稼働
- **Redis**: ローカルで稼働（localhost:6379）

### Docker環境確認
```bash
# 認証サーバーの JWKS エンドポイントにアクセス可能かテスト
curl -s http://host.docker.internal:8080/oauth/.well-known/jwks.json
# {"keys": [...]} が返ることを確認
```

### システム依存関係 (macOS)
```bash
# Homebrewが未インストールの場合はインストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 必要なパッケージのインストール
brew install python@3.11 postgresql redis git
```

---

## プロジェクトセットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd blead-stamp-svr
```

### 2. 仮想環境の作成
```bash
# venv環境の作成
python3 -m venv venv

# 環境のアクティベート (Linux/macOS)
source venv/bin/activate

# 環境のアクティベート (Windows)
# venv\Scripts\activate
```

### 3. 依存関係のインストール
```bash
# pipのアップグレード
pip install --upgrade pip

# プロジェクト依存関係のインストール
pip install -r requirements.txt
```

### 4. インストール確認
```bash
# Djangoインストール確認
python -c "import django; print(django.get_version())"

# その他の主要パッケージ確認
python -c "import rest_framework, redis, psycopg2; print('全パッケージが正常にインストールされました')"
```

---

## 環境設定

### 1. 環境設定テンプレートのコピー
```bash
cp .env.example .env
```

### 2. 環境変数の設定（Docker環境対応）
`.env`ファイルを編集して、あなたの環境に合わせて設定してください:

```bash
# Django設定
SECRET_KEY=django-insecure-replace-this-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# データベース設定（PostgreSQL）
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres

# Redis設定（JWKSキャッシュ・JWT検証キャッシュ用）
REDIS_URL=redis://localhost:6379/1

# 認証サーバー設定（Docker環境用）
AUTH_SERVER_URL=http://host.docker.internal:8080
JWKS_URL=http://host.docker.internal:8080/oauth/.well-known/jwks.json
JWT_ALGORITHM=RS256
JWT_AUDIENCE=bff-web-client
JWT_ISSUER=http://host.docker.internal:8080/oauth

# BFF-Web統合
BFF_WEB_URL=http://localhost:3000
CORS_ALLOWED_ORIGINS=http://localhost:3000

# Docker環境設定
DOCKER_ENVIRONMENT=False

# API設定
API_VERSION=v1
API_TITLE=REST API Server
API_DESCRIPTION=JWT authentication enabled REST API server for BFF-Web integration

# レート制限設定
THROTTLE_ANON=100/hour
THROTTLE_USER=1000/hour
THROTTLE_AUTH=60/min
THROTTLE_CONTENT_CREATE=10/hour
THROTTLE_CONTENT_LIKE=100/hour

# ログレベル
LOG_LEVEL=INFO
```

### 3. 認証サーバー接続確認
```bash
# 認証サーバーのJWKSエンドポイント接続確認
curl -s http://host.docker.internal:8080/oauth/.well-known/jwks.json

# 期待されるレスポンス例:
# {"keys": [{"kty": "RSA", "use": "sig", "kid": "1", "alg": "RS256", ...}]}
```

### 3. シークレットキーの生成 (本番環境用)
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

---

## データベースセットアップ

### 1. PostgreSQLセットアップ
```bash
# PostgreSQLサービスの開始
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# データベースとユーザーの作成
sudo -u postgres psql
```

```sql
-- データベースの作成
CREATE DATABASE stamp_db;

-- ユーザーの作成
CREATE USER stamp_user WITH PASSWORD 'your_password';

-- 権限の付与
GRANT ALL PRIVILEGES ON DATABASE stamp_db TO stamp_user;

-- PostgreSQLの終了
\q
```

### 2. 代替案: SQLite (開発環境のみ)
迅速な開発セットアップのため、SQLiteを使用する場合は`.env`を更新:
```bash
# PostgreSQL URLをコメントアウト
# DATABASE_URL=postgresql://...

# DjangoはデフォルトでSQLiteを使用します
```

### 3. マイグレーションの実行
```bash
# 仮想環境のアクティベート
source venv/bin/activate

# 問題のチェック
python manage.py check

# マイグレーションの実行
python manage.py migrate

# スーパーユーザーの作成
python manage.py createsuperuser
```

### 4. 静的ファイルの収集
```bash
python manage.py collectstatic --noinput
```

---

## 開発サーバー

### 1. Redisの開始 (必須)
```bash
# Linux
sudo systemctl start redis

# macOS
brew services start redis

# またはフォアグラウンドで実行
redis-server
```

### 2. Djangoサーバーの開始
```bash
# 仮想環境のアクティベート
source venv/bin/activate

# 開発サーバーの開始
python manage.py runserver 0.0.0.0:8000
```

### 3. サーバー動作確認
ブラウザで以下のURLにアクセスしてください:
- **管理画面**: http://localhost:8000/admin/
- **API仕様書**: http://localhost:8000/api/docs/
- **APIスキーマ**: http://localhost:8000/api/schema/

---

## VS Code設定

### 1. 必要な拡張機能のインストール
```bash
# VS Code拡張機能のインストール
code --install-extension ms-python.python
code --install-extension ms-python.debugpy
code --install-extension ms-python.pylance
```

### 2. VS Codeでプロジェクトを開く
```bash
code .
```

### 3. Pythonインタープリターの選択
1. `Ctrl+Shift+P` (macOSでは `Cmd+Shift+P`) を押す
2. "Python: Select Interpreter" と入力
3. `./venv/bin/python` を選択

### 4. デバッグ設定
プロジェクトには事前設定されたデバッグ設定が含まれています:
- **F5**を押してデバッグを開始
- **"Python: Django Server (venv)"** を選択
- デバッグ機能付きでサーバーが開始されます

### 5. 利用可能なデバッグ設定
- **Django Server (venv)**: 開発サーバーの起動
- **Django Shell (venv)**: インタラクティブなDjangoシェル
- **Django Tests (venv)**: テストスイートの実行
- **Django Migrate (venv)**: データベースマイグレーションの実行

---

## インストール確認

### 1. システムチェックの実行
```bash
python manage.py check --settings=config.settings
```

### 2. テストスイートの実行

**🔥 重要**: テスト時もPostgreSQLを使用します（SQLiteではありません）

```bash
# 全テストの実行（PostgreSQL使用）
python manage.py test

# 特定のアプリのテスト実行
python manage.py test stamp

# 詳細出力でのテスト実行
python manage.py test --verbosity=2

# テスト時のデータベース確認
python manage.py test --debug-mode
```

#### テストデータベース設定
- **テスト用DB名**: `test_postgres` 
- **エンジン**: PostgreSQL（本番と同じ）
- **自動作成**: テスト実行時に自動作成・削除
- **GIS機能**: PostgreSQLの数学関数を活用した位置情報検索（PostGIS不要）

### 3. APIエンドポイントのテスト
```bash
# ヘルスエンドポイントのテスト (認証が必要)
curl -H "Authorization: Bearer dummy-token" http://localhost:8000/api/v1/health/

# API仕様書のテスト
curl -I http://localhost:8000/api/docs/
```

### 4. 管理画面のテスト
1. http://localhost:8000/admin/ にアクセス
2. スーパーユーザーの認証情報でログイン
3. Django管理画面にアクセスできることを確認

---

## トラブルシューティング

### よくある問題

#### 1. ポートが既に使用中
```bash
# ポート8000を使用しているプロセスを検索
lsof -i :8000

# プロセスを終了
kill <PID>

# または別のポートを使用
python manage.py runserver 8001
```

#### 2. 仮想環境の問題
```bash
# 現在の環境を無効化
deactivate

# venvを削除して再作成
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. データベース接続エラー
```bash
# PostgreSQLが動作しているかチェック
sudo systemctl status postgresql

# データベースが存在するかチェック
sudo -u postgres psql -l

# 接続テスト
python manage.py dbshell
```

#### 4. Redis接続エラー
```bash
# Redisが動作しているかチェック
redis-cli ping

# "PONG"が返されるはずです

# Redisサービスの確認
sudo systemctl status redis
```

#### 5. 静的ファイルの問題
```bash
# 静的ファイルの再収集
python manage.py collectstatic --clear --noinput

# STATIC_ROOT設定の確認
python manage.py diffsettings | grep STATIC
```

#### 6. インポートエラー
```bash
# Pythonパスの確認
python -c "import sys; print(sys.path)"

# パッケージの再インストール
pip install --force-reinstall -r requirements.txt
```

### 環境固有の問題

#### Docker環境
Dockerで実行している場合は、`.env`を更新してください:
```bash
DOCKER_ENVIRONMENT=True
AUTH_SERVER_URL=http://host.docker.internal:8080
BFF_WEB_URL=http://host.docker.internal:3000
REDIS_URL=redis://host.docker.internal:6379/1
```

#### 本番環境
本番環境でのデプロイでは:
1. `DEBUG=False`に設定
2. 適切な`ALLOWED_HOSTS`を設定
3. 強力な`SECRET_KEY`を使用
4. 本番データベースを設定
5. 適切な静的ファイル配信を設定

### ヘルプの取得

#### ログの確認
```bash
# Djangoログ (設定されている場合)
tail -f logs/django.log

# PostgreSQLログ
sudo tail -f /var/log/postgresql/postgresql-*.log

# Redisログ
sudo tail -f /var/log/redis/redis-server.log
```

#### デバッグモード
```bash
# 最大詳細度で実行
python manage.py runserver --verbosity=3

# Djangoデバッグツールバーの有効化 (開発環境のみ)
# .envに追加: ENABLE_DEBUG_TOOLBAR=True
```

#### システム情報
```bash
# Pythonバージョン
python --version

# インストール済みパッケージ
pip list

# Djangoバージョン
python -c "import django; print(django.get_version())"

# システム情報
uname -a
```

---

## 次のステップ

インストール成功後:

1. **API仕様書を読む**: [03-api-specification.md](03-api-specification.md)
2. **認証を設定する**: [02-jwt-validation-guide.md](02-jwt-validation-guide.md)
3. **テストを設定する**: [08-testing-guide.md](08-testing-guide.md)
4. **監視を設定する**: [09-monitoring-guide.md](09-monitoring-guide.md)
5. **デプロイの準備**: [10-deployment-guide.md](10-deployment-guide.md)

---

## クイックリファレンス

### 必須コマンド
```bash
# 環境のアクティベート
source venv/bin/activate

# サーバー開始
python manage.py runserver

# マイグレーション実行
python manage.py migrate

# スーパーユーザー作成
python manage.py createsuperuser

# テスト実行
python manage.py test

# 静的ファイル収集
python manage.py collectstatic
```

### 主要URL
- 管理画面: http://localhost:8000/admin/
- API仕様書: http://localhost:8000/api/docs/
- APIスキーマ: http://localhost:8000/api/schema/
- ヘルスチェック: http://localhost:8000/api/v1/health/

### デフォルト認証情報
- **ユーザー名**: admin
- **メール**: admin@houwa-js.co.jp
- **パスワード**: admin

> ⚠️ **セキュリティ注意**: 本番環境ではデフォルト認証情報を変更してください。

---

*追加のヘルプについては、Djangoドキュメント https://docs.djangoproject.com/ またはDjango REST Frameworkドキュメント https://www.django-rest-framework.org/ を参照してください*