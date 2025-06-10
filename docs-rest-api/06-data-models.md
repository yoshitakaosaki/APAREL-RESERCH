# データモデル設計ガイド

## 1. 概要

REST APIサーバーにおけるビジネスロジック用データモデルの設計指針と実装例を提供します。認証情報は認証サーバーが管理するため、このAPIサーバーではビジネスデータのみを扱います。

## 2. 設計原則

### 2.1 基本原則

1. **認証分離**: 認証情報（パスワード、ログイン履歴等）は保持しない
2. **ユーザーID連携**: 認証サーバーのユーザーIDを外部キーとして使用
3. **監査証跡**: すべての重要な操作に対して監査ログを保持
4. **ソフトデリート**: 重要なデータは物理削除せず、論理削除を使用
5. **タイムゾーン**: UTCで統一して保存、表示時に変換

### 2.2 命名規則

```python
# モデル名: 単数形、パスカルケース
class UserProfile(models.Model):
    pass

# フィールド名: スネークケース
created_at = models.DateTimeField()

# 外部キー: _id サフィックス
user_id = models.CharField()  # 認証サーバーのユーザーID
category_id = models.ForeignKey()

# ブール値: is_ または has_ プレフィックス
is_active = models.BooleanField()
has_verified_email = models.BooleanField()
```

## 3. 基本モデル設計

### 3.1 抽象基底モデル

```python
# api/models/base.py
from django.db import models
import uuid
from django.utils import timezone

class BaseModel(models.Model):
    """すべてのモデルの基底クラス"""
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        db_index=True
    )
    
    class Meta:
        abstract = True
        ordering = ['-created_at']

class SoftDeleteModel(BaseModel):
    """論理削除をサポートする基底クラス"""
    
    is_deleted = models.BooleanField(
        default=False,
        db_index=True
    )
    deleted_at = models.DateTimeField(
        null=True,
        blank=True
    )
    deleted_by = models.CharField(
        max_length=255,
        null=True,
        blank=True
    )
    
    class Meta:
        abstract = True
    
    def soft_delete(self, user_id: str):
        """論理削除"""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.deleted_by = user_id
        self.save()
    
    def restore(self):
        """削除取り消し"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.save()

class AuditableModel(BaseModel):
    """監査可能な基底クラス"""
    
    created_by = models.CharField(
        max_length=255,
        db_index=True
    )
    updated_by = models.CharField(
        max_length=255
    )
    version = models.IntegerField(
        default=1
    )
    
    class Meta:
        abstract = True
    
    def save(self, *args, **kwargs):
        # バージョン管理
        if self.pk:
            self.version += 1
        super().save(*args, **kwargs)
```

### 3.2 カスタムマネージャー

```python
# api/models/managers.py
from django.db import models
from django.utils import timezone

class SoftDeleteManager(models.Manager):
    """論理削除を考慮したマネージャー"""
    
    def get_queryset(self):
        """削除されていないレコードのみ返す"""
        return super().get_queryset().filter(is_deleted=False)
    
    def deleted(self):
        """削除されたレコードのみ返す"""
        return super().get_queryset().filter(is_deleted=True)
    
    def with_deleted(self):
        """すべてのレコードを返す"""
        return super().get_queryset()

class PublishedManager(models.Manager):
    """公開済みコンテンツマネージャー"""
    
    def get_queryset(self):
        return super().get_queryset().filter(
            status='published',
            published_at__lte=timezone.now()
        )
```

## 4. ユーザー関連モデル

### 4.1 ユーザープロフィール

```python
# api/models/users.py
from django.db import models
from django.contrib.postgres.fields import JSONField
from .base import BaseModel, SoftDeleteModel
from .managers import SoftDeleteManager

class UserProfile(SoftDeleteModel):
    """
    ユーザープロフィール拡張
    認証情報は含まず、ビジネスロジックに必要な情報のみ保持
    """
    
    # 認証サーバーとの連携
    user_id = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text='認証サーバーのユーザーID'
    )
    
    # 基本情報（認証サーバーからキャッシュ）
    email = models.EmailField(
        db_index=True,
        help_text='認証サーバーから同期'
    )
    email_verified = models.BooleanField(
        default=False
    )
    
    # プロフィール情報
    display_name = models.CharField(
        max_length=255,
        blank=True
    )
    bio = models.TextField(
        blank=True,
        max_length=1000
    )
    avatar_url = models.URLField(
        blank=True
    )
    cover_image_url = models.URLField(
        blank=True
    )
    
    # 追加情報
    location = models.CharField(
        max_length=255,
        blank=True
    )
    website = models.URLField(
        blank=True
    )
    company = models.CharField(
        max_length=255,
        blank=True
    )
    job_title = models.CharField(
        max_length=255,
        blank=True
    )
    
    # ソーシャルリンク
    social_links = models.JSONField(
        default=dict,
        blank=True
    )
    
    # プリファレンス
    preferences = models.JSONField(
        default=dict
    )
    
    # 統計情報
    followers_count = models.IntegerField(
        default=0
    )
    following_count = models.IntegerField(
        default=0
    )
    posts_count = models.IntegerField(
        default=0
    )
    
    # メタデータ
    metadata = models.JSONField(
        default=dict,
        blank=True
    )
    
    # マネージャー
    objects = SoftDeleteManager()
    
    class Meta:
        db_table = 'user_profiles'
        indexes = [
            models.Index(fields=['user_id']),
            models.Index(fields=['email']),
            models.Index(fields=['display_name']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.display_name or self.email} ({self.user_id})"
    
    @property
    def language(self):
        return self.preferences.get('language', 'ja')
    
    @property
    def timezone(self):
        return self.preferences.get('timezone', 'Asia/Tokyo')
    
    @property
    def theme(self):
        return self.preferences.get('theme', 'light')
    
    def update_stats(self):
        """統計情報を更新"""
        # 実装例
        from .content import Content
        self.posts_count = Content.objects.filter(
            author_id=self.user_id
        ).count()
        self.save(update_fields=['posts_count', 'updated_at'])

class UserSettings(BaseModel):
    """ユーザー設定"""
    
    user_id = models.CharField(
        max_length=255,
        unique=True,
        db_index=True
    )
    
    # 通知設定
    notification_email = models.BooleanField(default=True)
    notification_push = models.BooleanField(default=False)
    notification_sms = models.BooleanField(default=False)
    
    # プライバシー設定
    profile_visibility = models.CharField(
        max_length=20,
        choices=[
            ('public', 'Public'),
            ('private', 'Private'),
            ('friends', 'Friends Only'),
        ],
        default='public'
    )
    show_email = models.BooleanField(default=False)
    show_location = models.BooleanField(default=True)
    
    # API設定
    api_rate_limit_override = models.IntegerField(
        null=True,
        blank=True,
        help_text='カスタムレート制限（管理者設定）'
    )
    
    class Meta:
        db_table = 'user_settings'
```

### 4.2 フォロー関係

```python
# api/models/social.py
from django.db import models
from .base import BaseModel

class Follow(BaseModel):
    """フォロー関係"""
    
    follower_id = models.CharField(
        max_length=255,
        db_index=True
    )
    following_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # フォロー状態
    is_mutual = models.BooleanField(
        default=False,
        help_text='相互フォローかどうか'
    )
    
    # 通知設定
    notify_posts = models.BooleanField(default=True)
    notify_mentions = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'follows'
        unique_together = ['follower_id', 'following_id']
        indexes = [
            models.Index(fields=['follower_id', '-created_at']),
            models.Index(fields=['following_id', '-created_at']),
            models.Index(fields=['is_mutual']),
        ]
    
    def save(self, *args, **kwargs):
        # 相互フォローチェック
        mutual = Follow.objects.filter(
            follower_id=self.following_id,
            following_id=self.follower_id
        ).exists()
        
        self.is_mutual = mutual
        super().save(*args, **kwargs)
        
        # 相手側も更新
        if mutual:
            Follow.objects.filter(
                follower_id=self.following_id,
                following_id=self.follower_id
            ).update(is_mutual=True)
```

## 5. コンテンツモデル

### 5.1 汎用コンテンツモデル

```python
# api/models/content.py
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils.text import slugify
from .base import SoftDeleteModel, AuditableModel
from .managers import SoftDeleteManager, PublishedManager
import markdown

class Content(SoftDeleteModel, AuditableModel):
    """汎用コンテンツモデル"""
    
    CONTENT_TYPES = [
        ('article', 'Article'),
        ('blog', 'Blog Post'),
        ('news', 'News'),
        ('page', 'Static Page'),
        ('document', 'Document'),
    ]
    
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('review', 'Under Review'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]
    
    # 基本情報
    content_type = models.CharField(
        max_length=20,
        choices=CONTENT_TYPES,
        default='article'
    )
    title = models.CharField(
        max_length=255,
        db_index=True
    )
    slug = models.SlugField(
        max_length=255,
        unique=True,
        db_index=True
    )
    
    # コンテンツ
    excerpt = models.TextField(
        blank=True,
        max_length=500,
        help_text='概要・抜粋'
    )
    content = models.TextField(
        help_text='Markdown形式'
    )
    content_html = models.TextField(
        blank=True,
        editable=False,
        help_text='レンダリング済みHTML'
    )
    
    # メタデータ
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='draft',
        db_index=True
    )
    author_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # カテゴリとタグ
    category = models.ForeignKey(
        'Category',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    tags = ArrayField(
        models.CharField(max_length=50),
        blank=True,
        default=list
    )
    
    # 公開設定
    published_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True
    )
    expires_at = models.DateTimeField(
        null=True,
        blank=True
    )
    
    # SEO
    seo_title = models.CharField(
        max_length=255,
        blank=True
    )
    seo_description = models.TextField(
        blank=True,
        max_length=160
    )
    seo_keywords = ArrayField(
        models.CharField(max_length=50),
        blank=True,
        default=list
    )
    
    # 統計
    views_count = models.IntegerField(default=0)
    likes_count = models.IntegerField(default=0)
    comments_count = models.IntegerField(default=0)
    shares_count = models.IntegerField(default=0)
    
    # 追加設定
    allow_comments = models.BooleanField(default=True)
    is_featured = models.BooleanField(default=False)
    is_pinned = models.BooleanField(default=False)
    
    # メタデータ
    metadata = models.JSONField(
        default=dict,
        blank=True
    )
    
    # マネージャー
    objects = SoftDeleteManager()
    published = PublishedManager()
    
    class Meta:
        db_table = 'contents'
        indexes = [
            models.Index(fields=['status', '-published_at']),
            models.Index(fields=['author_id', '-created_at']),
            models.Index(fields=['category', '-published_at']),
            models.Index(fields=['content_type', 'status']),
            models.Index(fields=['is_featured', '-published_at']),
            models.Index(fields=['tags']),
        ]
        ordering = ['-created_at']
    
    def save(self, *args, **kwargs):
        # スラッグ生成
        if not self.slug:
            self.slug = slugify(self.title)
        
        # Markdown -> HTML変換
        if self.content:
            md = markdown.Markdown(
                extensions=[
                    'markdown.extensions.extra',
                    'markdown.extensions.codehilite',
                    'markdown.extensions.toc',
                ]
            )
            self.content_html = md.convert(self.content)
        
        super().save(*args, **kwargs)
    
    @property
    def reading_time(self):
        """読了時間（分）"""
        word_count = len(self.content.split())
        return max(1, word_count // 200)
    
    @property
    def is_published(self):
        """公開中かどうか"""
        from django.utils import timezone
        now = timezone.now()
        
        if self.status != 'published':
            return False
        
        if self.published_at and self.published_at > now:
            return False
        
        if self.expires_at and self.expires_at < now:
            return False
        
        return True

class Category(BaseModel):
    """カテゴリ"""
    
    name = models.CharField(
        max_length=100,
        unique=True
    )
    slug = models.SlugField(
        max_length=100,
        unique=True
    )
    description = models.TextField(
        blank=True
    )
    parent = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='children'
    )
    
    # 表示設定
    is_active = models.BooleanField(default=True)
    display_order = models.IntegerField(default=0)
    
    # 統計
    content_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'categories'
        verbose_name_plural = 'Categories'
        ordering = ['display_order', 'name']
    
    def __str__(self):
        return self.name
```

### 5.2 コメントモデル

```python
# api/models/interactions.py
from django.db import models
from .base import SoftDeleteModel
from .managers import SoftDeleteManager

class Comment(SoftDeleteModel):
    """コメント"""
    
    # 対象
    content = models.ForeignKey(
        'Content',
        on_delete=models.CASCADE,
        related_name='comments'
    )
    
    # 投稿者
    author_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # コメント内容
    text = models.TextField(
        max_length=2000
    )
    
    # 親コメント（スレッド対応）
    parent = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='replies'
    )
    
    # 統計
    likes_count = models.IntegerField(default=0)
    replies_count = models.IntegerField(default=0)
    
    # 状態
    is_approved = models.BooleanField(
        default=True,
        help_text='承認済みかどうか'
    )
    is_spam = models.BooleanField(
        default=False,
        help_text='スパム判定'
    )
    
    # マネージャー
    objects = SoftDeleteManager()
    
    class Meta:
        db_table = 'comments'
        indexes = [
            models.Index(fields=['content', '-created_at']),
            models.Index(fields=['author_id', '-created_at']),
            models.Index(fields=['parent']),
        ]
        ordering = ['-created_at']
    
    def save(self, *args, **kwargs):
        # 返信数更新
        if self.parent_id and not self.pk:
            Comment.objects.filter(
                id=self.parent_id
            ).update(replies_count=models.F('replies_count') + 1)
        
        super().save(*args, **kwargs)

class Like(BaseModel):
    """いいね"""
    
    LIKE_TYPES = [
        ('content', 'Content'),
        ('comment', 'Comment'),
    ]
    
    # ユーザー
    user_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # 対象
    like_type = models.CharField(
        max_length=20,
        choices=LIKE_TYPES
    )
    object_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    class Meta:
        db_table = 'likes'
        unique_together = ['user_id', 'like_type', 'object_id']
        indexes = [
            models.Index(fields=['like_type', 'object_id']),
            models.Index(fields=['user_id', '-created_at']),
        ]
```

## 6. ファイル管理モデル

```python
# api/models/files.py
from django.db import models
from .base import BaseModel
import os

class FileUpload(BaseModel):
    """ファイルアップロード管理"""
    
    FILE_TYPES = [
        ('image', 'Image'),
        ('document', 'Document'),
        ('video', 'Video'),
        ('audio', 'Audio'),
        ('other', 'Other'),
    ]
    
    # アップロード者
    uploaded_by = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # ファイル情報
    file = models.FileField(
        upload_to='uploads/%Y/%m/%d/'
    )
    original_filename = models.CharField(
        max_length=255
    )
    file_type = models.CharField(
        max_length=20,
        choices=FILE_TYPES
    )
    mime_type = models.CharField(
        max_length=100
    )
    file_size = models.BigIntegerField()
    
    # 画像の場合
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    
    # メタデータ
    alt_text = models.CharField(
        max_length=255,
        blank=True
    )
    description = models.TextField(
        blank=True
    )
    
    # アクセス制御
    is_public = models.BooleanField(default=False)
    access_count = models.IntegerField(default=0)
    
    # ハッシュ（重複チェック用）
    file_hash = models.CharField(
        max_length=64,
        db_index=True
    )
    
    class Meta:
        db_table = 'file_uploads'
        indexes = [
            models.Index(fields=['uploaded_by', '-created_at']),
            models.Index(fields=['file_type', '-created_at']),
            models.Index(fields=['file_hash']),
        ]
    
    def save(self, *args, **kwargs):
        # ファイルサイズ取得
        if self.file and not self.file_size:
            self.file_size = self.file.size
        
        # オリジナルファイル名保存
        if self.file and not self.original_filename:
            self.original_filename = os.path.basename(self.file.name)
        
        super().save(*args, **kwargs)
    
    @property
    def file_url(self):
        """ファイルURLを取得"""
        if self.file:
            return self.file.url
        return None
    
    def increment_access_count(self):
        """アクセス数をインクリメント"""
        self.access_count = models.F('access_count') + 1
        self.save(update_fields=['access_count'])
```

## 7. 通知モデル

```python
# api/models/notifications.py
from django.db import models
from .base import BaseModel

class Notification(BaseModel):
    """通知"""
    
    NOTIFICATION_TYPES = [
        ('follow', 'New Follower'),
        ('like', 'Like'),
        ('comment', 'Comment'),
        ('mention', 'Mention'),
        ('system', 'System'),
    ]
    
    # 受信者
    recipient_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # 通知タイプ
    notification_type = models.CharField(
        max_length=20,
        choices=NOTIFICATION_TYPES
    )
    
    # 送信者（システム通知の場合はnull）
    sender_id = models.CharField(
        max_length=255,
        null=True,
        blank=True
    )
    
    # 通知内容
    title = models.CharField(max_length=255)
    message = models.TextField()
    
    # リンク先
    link_url = models.CharField(
        max_length=500,
        blank=True
    )
    
    # 関連オブジェクト
    related_object_type = models.CharField(
        max_length=50,
        blank=True
    )
    related_object_id = models.CharField(
        max_length=255,
        blank=True
    )
    
    # 状態
    is_read = models.BooleanField(
        default=False,
        db_index=True
    )
    read_at = models.DateTimeField(
        null=True,
        blank=True
    )
    
    # 配信状態
    email_sent = models.BooleanField(default=False)
    push_sent = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'notifications'
        indexes = [
            models.Index(fields=['recipient_id', 'is_read', '-created_at']),
            models.Index(fields=['notification_type', '-created_at']),
        ]
        ordering = ['-created_at']
    
    def mark_as_read(self):
        """既読にする"""
        from django.utils import timezone
        self.is_read = True
        self.read_at = timezone.now()
        self.save(update_fields=['is_read', 'read_at'])
```

## 8. 分析用モデル

```python
# api/models/analytics.py
from django.db import models
from .base import BaseModel

class PageView(BaseModel):
    """ページビュー記録"""
    
    # セッション情報
    session_id = models.CharField(
        max_length=255,
        db_index=True
    )
    user_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        db_index=True
    )
    
    # ページ情報
    path = models.CharField(
        max_length=500,
        db_index=True
    )
    referrer = models.CharField(
        max_length=500,
        blank=True
    )
    
    # デバイス情報
    user_agent = models.TextField()
    ip_address = models.GenericIPAddressField()
    
    # 地理情報
    country_code = models.CharField(
        max_length=2,
        blank=True
    )
    region = models.CharField(
        max_length=100,
        blank=True
    )
    
    # パフォーマンス
    load_time_ms = models.IntegerField(
        null=True,
        blank=True
    )
    
    class Meta:
        db_table = 'page_views'
        indexes = [
            models.Index(fields=['path', '-created_at']),
            models.Index(fields=['user_id', '-created_at']),
            models.Index(fields=['session_id']),
            models.Index(fields=['-created_at']),
        ]

class UserActivity(BaseModel):
    """ユーザーアクティビティ"""
    
    ACTIVITY_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('create', 'Create'),
        ('update', 'Update'),
        ('delete', 'Delete'),
        ('share', 'Share'),
        ('export', 'Export'),
    ]
    
    # ユーザー
    user_id = models.CharField(
        max_length=255,
        db_index=True
    )
    
    # アクティビティ
    activity_type = models.CharField(
        max_length=20,
        choices=ACTIVITY_TYPES
    )
    description = models.CharField(
        max_length=500
    )
    
    # 対象
    target_type = models.CharField(
        max_length=50,
        blank=True
    )
    target_id = models.CharField(
        max_length=255,
        blank=True
    )
    
    # メタデータ
    metadata = models.JSONField(
        default=dict,
        blank=True
    )
    
    class Meta:
        db_table = 'user_activities'
        indexes = [
            models.Index(fields=['user_id', '-created_at']),
            models.Index(fields=['activity_type', '-created_at']),
        ]
```

## 9. マイグレーション戦略

### 9.1 初期マイグレーション

```bash
# マイグレーション作成
python manage.py makemigrations api

# SQLプレビュー
python manage.py sqlmigrate api 0001

# マイグレーション実行
python manage.py migrate

# カスタムマイグレーション作成
python manage.py makemigrations api --empty -n add_indexes
```

### 9.2 データマイグレーション例

```python
# api/migrations/0002_populate_initial_data.py
from django.db import migrations

def create_default_categories(apps, schema_editor):
    Category = apps.get_model('api', 'Category')
    
    categories = [
        {'name': '技術', 'slug': 'tech'},
        {'name': 'ビジネス', 'slug': 'business'},
        {'name': 'ライフスタイル', 'slug': 'lifestyle'},
    ]
    
    for cat_data in categories:
        Category.objects.create(**cat_data)

def reverse_categories(apps, schema_editor):
    Category = apps.get_model('api', 'Category')
    Category.objects.all().delete()

class Migration(migrations.Migration):
    dependencies = [
        ('api', '0001_initial'),
    ]
    
    operations = [
        migrations.RunPython(
            create_default_categories,
            reverse_categories
        ),
    ]
```

## 10. パフォーマンス最適化

### 10.1 インデックス戦略

```python
# 複合インデックスの例
class Meta:
    indexes = [
        # 頻繁なクエリパターンに基づくインデックス
        models.Index(
            fields=['status', 'published_at'],
            name='idx_content_status_published'
        ),
        
        # 部分インデックス（PostgreSQL）
        models.Index(
            fields=['created_at'],
            name='idx_active_content',
            condition=models.Q(is_deleted=False)
        ),
    ]
```

### 10.2 クエリ最適化

```python
# select_related / prefetch_related の使用
contents = Content.objects.filter(
    status='published'
).select_related(
    'category'
).prefetch_related(
    'comments',
    'comments__author'
)

# アノテーションの使用
from django.db.models import Count, Avg

contents_with_stats = Content.objects.annotate(
    comment_count=Count('comments'),
    avg_rating=Avg('ratings__score')
)
```

## まとめ

このデータモデル設計により、認証サーバーと適切に分離されたビジネスロジックを実装できます。重要なポイント：

1. **認証分離**: 認証情報は保持せず、user_id での連携
2. **拡張性**: 基底クラスによる共通機能の提供
3. **監査性**: すべての操作の追跡可能性
4. **パフォーマンス**: 適切なインデックスとクエリ最適化
5. **保守性**: 明確な命名規則と構造