# utils/file_security.py - Enhanced File Security and Validation
import os
import hashlib
import mimetypes
from pathlib import Path

# Optional magic import with fallback
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    import warnings
    warnings.warn("python-magic not available. File type detection will use basic mimetypes only.")
    magic = None
from typing import Dict, List, Tuple, Optional
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

class FileSecurityValidator:
    """Enhanced file security validation"""
    
    # Safe MIME types for each extension
    SAFE_MIME_TYPES = {
        'csv': ['text/csv', 'text/plain', 'application/csv'],
        'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
        'xls': ['application/vnd.ms-excel'],
        'json': ['application/json', 'text/json', 'text/plain'],
        'txt': ['text/plain'],
        'zip': ['application/zip']
    }
    
    # Maximum file sizes by type (in bytes)
    MAX_FILE_SIZES = {
        'csv': 500 * 1024 * 1024,   # 500MB
        'xlsx': 100 * 1024 * 1024,  # 100MB
        'xls': 50 * 1024 * 1024,    # 50MB
        'json': 50 * 1024 * 1024,   # 50MB
        'txt': 10 * 1024 * 1024,    # 10MB
        'zip': 200 * 1024 * 1024    # 200MB
    }
    
    # Dangerous file signatures (magic bytes)
    DANGEROUS_SIGNATURES = [
        b'\x4d\x5a',  # PE executable
        b'\x7f\x45\x4c\x46',  # ELF executable
        b'\xca\xfe\xba\xbe',  # Java class file
        b'\xfe\xed\xfa',  # Mach-O binary
        b'#!/bin/',  # Shell script
        b'#!/usr/bin/',  # Shell script
        b'<?php',  # PHP script
        b'<script',  # JavaScript in HTML
    ]
    
    def __init__(self):
        self.quarantine_dir = Path("quarantine")
        self.quarantine_dir.mkdir(exist_ok=True)
    
    def validate_filename(self, filename: str) -> Tuple[bool, str, str]:
        """
        Validate and sanitize filename
        Returns: (is_valid, sanitized_filename, error_message)
        """
        if not filename:
            return False, "", "Empty filename"
        
        # Get original extension
        original_ext = Path(filename).suffix.lower()
        
        # Sanitize filename
        sanitized = secure_filename(filename)
        if not sanitized:
            return False, "", "Invalid filename after sanitization"
        
        # Check for double extensions (potential security risk)
        if sanitized.count('.') > 1:
            # Allow only specific double extensions
            allowed_double_ext = ['.tar.gz', '.tar.bz2']
            if not any(sanitized.lower().endswith(ext) for ext in allowed_double_ext):
                return False, sanitized, "Multiple file extensions not allowed"
        
        # Ensure extension is preserved and allowed
        ext = Path(sanitized).suffix.lower().lstrip('.')
        if ext not in self.SAFE_MIME_TYPES:
            return False, sanitized, f"File extension '{ext}' not allowed"
        
        return True, sanitized, ""
    
    def validate_file_content(self, file_path: Path) -> Tuple[bool, Dict[str, any]]:
        """
        Validate file content using multiple methods
        Returns: (is_safe, validation_info)
        """
        validation_info = {
            'file_size': 0,
            'mime_type': None,
            'file_signature': None,
            'extension': None,
            'hash_md5': None,
            'is_text_file': False,
            'encoding': None
        }
        
        try:
            # Basic file info
            stat_info = file_path.stat()
            validation_info['file_size'] = stat_info.st_size
            validation_info['extension'] = file_path.suffix.lower().lstrip('.')
            
            # Check file size limits
            max_size = self.MAX_FILE_SIZES.get(validation_info['extension'], 10 * 1024 * 1024)
            if validation_info['file_size'] > max_size:
                logger.warning(f"File {file_path.name} exceeds size limit: {validation_info['file_size']} > {max_size}")
                return False, validation_info
            
            # Read file signature (first 512 bytes)
            with open(file_path, 'rb') as f:
                file_signature = f.read(512)
                validation_info['file_signature'] = file_signature[:20].hex()
            
            # Check for dangerous signatures
            for dangerous_sig in self.DANGEROUS_SIGNATURES:
                if file_signature.startswith(dangerous_sig):
                    logger.error(f"Dangerous file signature detected in {file_path.name}: {dangerous_sig.hex()}")
                    return False, validation_info
            
            # Use python-magic for MIME type detection if available
            if MAGIC_AVAILABLE and magic:
                try:
                    validation_info['mime_type'] = magic.from_file(str(file_path), mime=True)
                except Exception as e:
                    # Fallback to mimetypes module
                    validation_info['mime_type'], _ = mimetypes.guess_type(str(file_path))
                    logger.debug(f"python-magic failed, using mimetypes: {e}")
            else:
                # Use mimetypes module as fallback
                validation_info['mime_type'], _ = mimetypes.guess_type(str(file_path))
                logger.debug("Using mimetypes for MIME type detection (python-magic not available)")
            
            # Validate MIME type matches extension
            expected_mimes = self.SAFE_MIME_TYPES.get(validation_info['extension'], [])
            if validation_info['mime_type'] not in expected_mimes:
                logger.warning(f"MIME type mismatch for {file_path.name}: got {validation_info['mime_type']}, expected one of {expected_mimes}")
                return False, validation_info
            
            # Calculate MD5 hash for file integrity
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            validation_info['hash_md5'] = hash_md5.hexdigest()
            
            # Check if file is text-based (for CSV/JSON validation)
            if validation_info['extension'] in ['csv', 'json', 'txt']:
                try:
                    import chardet
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(1024)
                    encoding_result = chardet.detect(raw_data)
                    validation_info['encoding'] = encoding_result['encoding']
                    validation_info['is_text_file'] = encoding_result['confidence'] > 0.7
                    
                    if not validation_info['is_text_file']:
                        logger.warning(f"File {file_path.name} appears to be binary but has text extension")
                        return False, validation_info
                        
                except ImportError:
                    logger.warning("chardet not available, skipping encoding detection")
            
            return True, validation_info
            
        except Exception as e:
            logger.error(f"Error validating file content {file_path.name}: {e}")
            return False, validation_info
    
    def quarantine_file(self, file_path: Path, reason: str) -> Path:
        """Move suspicious file to quarantine"""
        quarantine_path = self.quarantine_dir / f"{file_path.name}_{int(os.time())}"
        try:
            file_path.rename(quarantine_path)
            logger.warning(f"File quarantined: {file_path.name} -> {quarantine_path} (Reason: {reason})")
            return quarantine_path
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path.name}: {e}")
            raise
    
    def validate_upload(self, file, upload_dir: Path) -> Tuple[bool, Path, Dict[str, any]]:
        """
        Complete file upload validation process
        Returns: (is_safe, file_path, validation_info)
        """
        # Step 1: Validate filename
        is_valid, sanitized_filename, error = self.validate_filename(file.filename)
        if not is_valid:
            raise ValueError(f"Invalid filename: {error}")
        
        # Step 2: Save file temporarily
        temp_path = upload_dir / sanitized_filename
        counter = 1
        while temp_path.exists():
            name, ext = os.path.splitext(sanitized_filename)
            temp_path = upload_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        try:
            file.save(str(temp_path))
            
            # Step 3: Validate file content
            is_safe, validation_info = self.validate_file_content(temp_path)
            
            if not is_safe:
                # Quarantine the file
                quarantine_path = self.quarantine_file(temp_path, "Failed content validation")
                raise ValueError(f"File failed security validation and has been quarantined")
            
            logger.info(f"File validation successful: {sanitized_filename}")
            return True, temp_path, validation_info
            
        except Exception as e:
            # Clean up temp file if validation fails
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise