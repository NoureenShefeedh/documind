import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import sessionmaker

from .auth import AuthError, create_token, decode_token, hash_password, verify_password
from .config import Settings, get_settings
from .db import PgRepository, init_db, make_engine
from .rag.embeddings import Embedder, get_embedder
from .rag.llm import Answerer, LLMError, get_answerer
from .rag.loaders import DocumentParseError, UnsupportedFileTypeError
from .rag.pipeline import RagService
from .repository import EmailTakenError, Repository
from .schemas import Credentials, DocumentOut, QueryIn, QueryOut, SourceOut, TokenOut

log = logging.getLogger("documind")
bearer = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None,
    repo: Repository | None = None,
    embedder: Embedder | None = None,
    answerer: Answerer | None = None,
) -> FastAPI:
    """App factory. Pass ``repo`` (e.g. MemoryRepository) to run without Postgres."""
    settings = settings or get_settings()
    if settings.jwt_secret == "change-me":
        log.warning("JWT_SECRET is the default value; set a strong secret in production")
    embedder = embedder or get_embedder(settings)
    answerer = answerer or get_answerer(settings)

    def build(repository: Repository) -> None:
        app.state.repo = repository
        app.state.service = RagService(
            repository,
            embedder,
            answerer,
            chunk_words=settings.chunk_words,
            chunk_overlap=settings.chunk_overlap,
            default_top_k=settings.top_k,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine = None
        if repo is None:
            engine = make_engine(settings.database_url)
            init_db(engine)
            build(PgRepository(sessionmaker(engine, expire_on_commit=False)))
        yield
        if engine is not None:
            engine.dispose()

    app = FastAPI(title="DocuMind", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if repo is not None:
        build(repo)

    def current_user_id(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> int:
        unauthorized = HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        if creds is None:
            raise unauthorized
        try:
            user_id = decode_token(creds.credentials, settings.jwt_secret, settings.jwt_algorithm)
        except AuthError:
            raise unauthorized from None
        if app.state.repo.get_user(user_id) is None:
            raise unauthorized
        return user_id

    def token_for(user_id: int) -> TokenOut:
        return TokenOut(
            access_token=create_token(
                user_id, settings.jwt_secret, settings.jwt_expire_minutes, settings.jwt_algorithm
            )
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/auth/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
    def register(body: Credentials) -> TokenOut:
        try:
            user = app.state.repo.create_user(body.email.lower(), hash_password(body.password))
        except EmailTakenError:
            raise HTTPException(status.HTTP_409_CONFLICT, "email already registered") from None
        return token_for(user.id)

    @app.post("/auth/login", response_model=TokenOut)
    def login(body: Credentials) -> TokenOut:
        user = app.state.repo.get_user_by_email(body.email.lower())
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
        return token_for(user.id)

    @app.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
    def upload(
        file: UploadFile = File(...), user_id: int = Depends(current_user_id)
    ) -> DocumentOut:
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds {settings.max_upload_mb} MB",
            )
        try:
            doc = app.state.service.ingest(user_id, file.filename or "untitled.txt", data)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None
        except DocumentParseError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
        return DocumentOut(
            id=doc.id, filename=doc.filename, n_chunks=doc.n_chunks, created_at=doc.created_at
        )

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(user_id: int = Depends(current_user_id)) -> list[DocumentOut]:
        return [
            DocumentOut(id=d.id, filename=d.filename, n_chunks=d.n_chunks, created_at=d.created_at)
            for d in app.state.repo.list_documents(user_id)
        ]

    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(document_id: int, user_id: int = Depends(current_user_id)) -> None:
        if not app.state.repo.delete_document(user_id, document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")

    @app.post("/query", response_model=QueryOut)
    def query(body: QueryIn, user_id: int = Depends(current_user_id)) -> QueryOut:
        try:
            result = app.state.service.ask(user_id, body.question, body.top_k)
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "answer generation failed") from None
        return QueryOut(
            answer=result.answer,
            sources=[
                SourceOut(
                    document=s.document,
                    chunk_index=s.chunk_index,
                    text=s.text,
                    score=s.score,
                )
                for s in result.sources
            ],
        )

    return app
