# libs/core/functions/img_processor.py
"""
이미지 처리 모듈

이미지 데이터를 로컬 파일 시스템에 저장하고 태그 형식으로 변환하는 기능을 제공합니다.
기존 이미지 업로드 함수를 대체하는 범용적인 이미지 처리 모듈입니다.

주요 기능:
- 이미지 데이터를 지정된 디렉토리에 저장
- 저장된 경로를 커스텀 태그 형식으로 반환
- 중복 이미지 감지 및 처리
- 다양한 이미지 포맷 지원

사용 예시:
    from libs.core.functions.img_processor import ImageProcessor

    # 기본 설정으로 사용
    processor = ImageProcessor()
    tag = processor.save_image(image_bytes)
    # 결과: "[Image:temp/abc123.png]"

    # 커스텀 설정
    processor = ImageProcessor(
        directory_path="output/images",
        tag_prefix="<img src='",
        tag_suffix="'>"
    )
    tag = processor.save_image(image_bytes)
    # 결과: "<img src='output/images/abc123.png'>"
"""
import hashlib
import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("document-processor")


class ImageFormat(Enum):
    """지원하는 이미지 포맷"""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class NamingStrategy(Enum):
    """이미지 파일 이름 생성 전략"""
    HASH = "hash"           # 이미지 내용 기반 해시 (중복 방지)
    UUID = "uuid"           # 고유 UUID
    SEQUENTIAL = "sequential"  # 순차 번호
    TIMESTAMP = "timestamp"    # 타임스탬프 기반


@dataclass
class ImageProcessorConfig:
    """
    ImageProcessor 설정

    Attributes:
        directory_path: 이미지를 저장할 디렉토리 경로
        tag_prefix: 태그 접두사 (예: "[Image:")
        tag_suffix: 태그 접미사 (예: "]")
        naming_strategy: 파일 이름 생성 전략
        default_format: 기본 이미지 포맷
        create_directory: 디렉토리가 없을 때 자동 생성 여부
        use_absolute_path: 태그에 절대 경로 사용 여부
        hash_algorithm: 해시 알고리즘 (hash 전략 시 사용)
        max_filename_length: 최대 파일 이름 길이
    """
    directory_path: str = "temp"
    tag_prefix: str = "[Image:"
    tag_suffix: str = "]"
    naming_strategy: NamingStrategy = NamingStrategy.HASH
    default_format: ImageFormat = ImageFormat.PNG
    create_directory: bool = True
    use_absolute_path: bool = False
    hash_algorithm: str = "sha256"
    max_filename_length: int = 64


class ImageProcessor:
    """
    이미지 처리 클래스

    이미지 데이터를 로컬 파일 시스템에 저장하고,
    저장된 경로를 지정된 태그 형식으로 반환합니다.

    Args:
        directory_path: 이미지 저장 디렉토리 (기본값: "temp")
        tag_prefix: 태그 접두사 (기본값: "[Image:")
        tag_suffix: 태그 접미사 (기본값: "]")
        naming_strategy: 파일 이름 생성 전략 (기본값: HASH)
        config: ImageProcessorConfig 객체 (개별 파라미터보다 우선)

    Examples:
        >>> processor = ImageProcessor()
        >>> tag = processor.save_image(image_bytes)
        "[Image:temp/a1b2c3d4.png]"

        >>> processor = ImageProcessor(
        ...     directory_path="images",
        ...     tag_prefix="![image](",
        ...     tag_suffix=")"
        ... )
        >>> tag = processor.save_image(image_bytes)
        "![image](images/a1b2c3d4.png)"
    """

    def __init__(
        self,
        directory_path: str = "temp",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        naming_strategy: Union[NamingStrategy, str] = NamingStrategy.HASH,
        config: Optional[ImageProcessorConfig] = None,
    ):
        if config:
            self.config = config
        else:
            # naming_strategy가 문자열인 경우 Enum으로 변환
            if isinstance(naming_strategy, str):
                naming_strategy = NamingStrategy(naming_strategy.lower())

            self.config = ImageProcessorConfig(
                directory_path=directory_path,
                tag_prefix=tag_prefix,
                tag_suffix=tag_suffix,
                naming_strategy=naming_strategy,
            )

        # 처리된 이미지 해시 추적 (중복 방지)
        self._processed_hashes: Dict[str, str] = {}

        # 순차 카운터 (sequential 전략용)
        self._sequential_counter: int = 0

        # 디렉토리 생성
        if self.config.create_directory:
            self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """디렉토리가 존재하는지 확인하고 없으면 생성"""
        path = Path(self.config.directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

    def _compute_hash(self, data: bytes) -> str:
        """이미지 데이터의 해시 계산"""
        hasher = hashlib.new(self.config.hash_algorithm)
        hasher.update(data)
        return hasher.hexdigest()[:16]  # 처음 16자만 사용

    def _detect_format(self, data: bytes) -> ImageFormat:
        """이미지 데이터에서 포맷 감지"""
        # 매직 바이트로 포맷 감지
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ImageFormat.PNG
        elif data[:2] == b'\xff\xd8':
            return ImageFormat.JPEG
        elif data[:6] in (b'GIF87a', b'GIF89a'):
            return ImageFormat.GIF
        elif data[:2] == b'BM':
            return ImageFormat.BMP
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return ImageFormat.WEBP
        elif data[:4] in (b'II*\x00', b'MM\x00*'):
            return ImageFormat.TIFF
        else:
            return ImageFormat.UNKNOWN

    def _generate_filename(
        self,
        data: bytes,
        image_format: ImageFormat,
        custom_name: Optional[str] = None
    ) -> str:
        """파일 이름 생성"""
        if custom_name:
            # 확장자 추가
            if not any(custom_name.lower().endswith(f".{fmt.value}") for fmt in ImageFormat if fmt != ImageFormat.UNKNOWN):
                ext = image_format.value if image_format != ImageFormat.UNKNOWN else self.config.default_format.value
                return f"{custom_name}.{ext}"
            return custom_name

        ext = image_format.value if image_format != ImageFormat.UNKNOWN else self.config.default_format.value

        strategy = self.config.naming_strategy

        if strategy == NamingStrategy.HASH:
            base = self._compute_hash(data)
        elif strategy == NamingStrategy.UUID:
            base = str(uuid.uuid4())[:16]
        elif strategy == NamingStrategy.SEQUENTIAL:
            self._sequential_counter += 1
            base = f"image_{self._sequential_counter:06d}"
        elif strategy == NamingStrategy.TIMESTAMP:
            import time
            base = f"img_{int(time.time() * 1000)}"
        else:
            base = self._compute_hash(data)

        filename = f"{base}.{ext}"

        # 파일 이름 길이 제한
        if len(filename) > self.config.max_filename_length:
            max_base_len = self.config.max_filename_length - len(ext) - 1
            filename = f"{base[:max_base_len]}.{ext}"

        return filename

    def _build_tag(self, file_path: str) -> str:
        """저장된 파일 경로로 태그 생성"""
        if self.config.use_absolute_path:
            path_str = str(Path(file_path).absolute())
        else:
            path_str = file_path

        # 경로 구분자 통일 (Windows -> Unix 스타일)
        path_str = path_str.replace("\\", "/")

        return f"{self.config.tag_prefix}{path_str}{self.config.tag_suffix}"

    def save_image(
        self,
        image_data: bytes,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        skip_duplicate: bool = True,
    ) -> Optional[str]:
        """
        이미지 데이터를 파일로 저장하고 태그를 반환합니다.

        Args:
            image_data: 이미지 바이너리 데이터
            custom_name: 커스텀 파일 이름 (확장자 제외 가능)
            processed_images: 처리된 이미지 경로 집합 (외부 중복 추적용)
            skip_duplicate: True이면 중복 이미지 저장 생략 (기존 경로 반환)

        Returns:
            이미지 태그 문자열, 실패 시 None

        Examples:
            >>> processor = ImageProcessor()
            >>> tag = processor.save_image(png_bytes)
            "[Image:temp/abc123.png]"
        """
        if not image_data:
            logger.warning("Empty image data provided")
            return None

        try:
            # 이미지 포맷 감지
            image_format = self._detect_format(image_data)

            # 해시 계산 (중복 체크용)
            image_hash = self._compute_hash(image_data)

            # 중복 체크
            if skip_duplicate and image_hash in self._processed_hashes:
                existing_path = self._processed_hashes[image_hash]
                logger.debug(f"Duplicate image detected, returning existing: {existing_path}")
                return self._build_tag(existing_path)

            # 파일 이름 생성
            filename = self._generate_filename(image_data, image_format, custom_name)

            # 전체 경로
            file_path = os.path.join(self.config.directory_path, filename)

            # 외부 중복 추적 체크
            if processed_images is not None and file_path in processed_images:
                logger.debug(f"Image already processed externally: {file_path}")
                return self._build_tag(file_path)

            # 디렉토리 확인
            self._ensure_directory_exists()

            # 파일 저장
            with open(file_path, 'wb') as f:
                f.write(image_data)

            logger.debug(f"Image saved: {file_path}")

            # 내부 중복 추적 업데이트
            self._processed_hashes[image_hash] = file_path

            # 외부 중복 추적 업데이트
            if processed_images is not None:
                processed_images.add(file_path)

            return self._build_tag(file_path)

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None

    def save_image_from_pil(
        self,
        pil_image,
        image_format: Optional[ImageFormat] = None,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        quality: int = 95,
    ) -> Optional[str]:
        """
        PIL Image 객체를 파일로 저장하고 태그를 반환합니다.

        Args:
            pil_image: PIL Image 객체
            image_format: 저장할 이미지 포맷 (None이면 원본 유지 또는 기본값)
            custom_name: 커스텀 파일 이름
            processed_images: 처리된 이미지 경로 집합
            quality: JPEG 품질 (1-100)

        Returns:
            이미지 태그 문자열, 실패 시 None
        """
        try:
            from PIL import Image

            if not isinstance(pil_image, Image.Image):
                logger.error("Invalid PIL Image object")
                return None

            # 포맷 결정
            fmt = image_format or ImageFormat.PNG
            if fmt == ImageFormat.UNKNOWN:
                fmt = self.config.default_format

            # 바이트로 변환
            buffer = io.BytesIO()
            save_format = fmt.value.upper()
            if save_format == "JPG":
                save_format = "JPEG"

            save_kwargs = {}
            if save_format == "JPEG":
                save_kwargs["quality"] = quality
            elif save_format == "PNG":
                save_kwargs["compress_level"] = 6

            pil_image.save(buffer, format=save_format, **save_kwargs)
            image_data = buffer.getvalue()

            return self.save_image(image_data, custom_name, processed_images)

        except Exception as e:
            logger.error(f"Failed to save PIL image: {e}")
            return None

    def get_processed_count(self) -> int:
        """처리된 이미지 수 반환"""
        return len(self._processed_hashes)

    def get_processed_paths(self) -> List[str]:
        """처리된 모든 이미지 경로 반환"""
        return list(self._processed_hashes.values())

    def clear_cache(self) -> None:
        """내부 중복 추적 캐시 초기화"""
        self._processed_hashes.clear()
        self._sequential_counter = 0

    def cleanup(self, delete_files: bool = False) -> int:
        """
        리소스 정리

        Args:
            delete_files: True이면 저장된 파일도 삭제

        Returns:
            삭제된 파일 수
        """
        deleted = 0

        if delete_files:
            for path in self._processed_hashes.values():
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete file {path}: {e}")

        self.clear_cache()
        return deleted


# ============================================================================
# 편의 함수
# ============================================================================

# 전역 기본 프로세서 (싱글톤 패턴)
_default_processor: Optional[ImageProcessor] = None


def get_default_processor(
    directory_path: str = "temp",
    tag_prefix: str = "[Image:",
    tag_suffix: str = "]",
) -> ImageProcessor:
    """
    기본 ImageProcessor 인스턴스를 반환합니다.

    최초 호출 시 생성되며, 이후에는 동일한 인스턴스를 반환합니다.
    다른 설정이 필요하면 새로운 ImageProcessor를 직접 생성하세요.
    """
    global _default_processor

    if _default_processor is None:
        _default_processor = ImageProcessor(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
        )

    return _default_processor


def save_image_to_file(
    image_data: bytes,
    directory_path: str = "temp",
    tag_prefix: str = "[Image:",
    tag_suffix: str = "]",
    processed_images: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    이미지를 파일로 저장하고 태그를 반환합니다.

    기존 이미지 업로드 함수를 대체하는 간단한 함수입니다.

    Args:
        image_data: 이미지 바이너리 데이터
        directory_path: 저장 디렉토리
        tag_prefix: 태그 접두사
        tag_suffix: 태그 접미사
        processed_images: 중복 추적용 집합

    Returns:
        이미지 태그 문자열, 실패 시 None

    Examples:
        >>> tag = save_image_to_file(image_bytes)
        "[Image:temp/abc123.png]"

        >>> tag = save_image_to_file(
        ...     image_bytes,
        ...     directory_path="output",
        ...     tag_prefix="<img src='",
        ...     tag_suffix="'/>"
        ... )
        "<img src='output/abc123.png'/>"
    """
    processor = ImageProcessor(
        directory_path=directory_path,
        tag_prefix=tag_prefix,
        tag_suffix=tag_suffix,
    )

    return processor.save_image(image_data, processed_images=processed_images)


__all__ = [
    # 클래스
    "ImageProcessor",
    "ImageProcessorConfig",
    "ImageFormat",
    "NamingStrategy",
    # 편의 함수
    "save_image_to_file",
    "get_default_processor",
]
