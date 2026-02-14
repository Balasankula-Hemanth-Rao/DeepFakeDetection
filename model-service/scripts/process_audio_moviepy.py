
import os
from pathlib import Path
# Fix for MoviePy 2.0
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError:
        # Fallback or error
        import logging
        logging.getLogger(__name__).error("Could not import VideoFileClip")
        raise

from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio_from_metadata(metadata_path, project_root):
    metadata_path = Path(metadata_path)
    project_root = Path(project_root)
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    videos = metadata.get('videos', [])
    logger.info(f"Loaded metadata for {len(videos)} videos")
    
    # Filter for 'real' videos only as requested (or process all if needed)
    # The user task specifically mentioned processing 'real' data because it was missing.
    # But checking 'fake' might be good too if missing. For now, focus on 'real'.
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for video_in_meta in tqdm(videos, desc="Processing Audio"):
        if video_in_meta.get('class') != 'real':
            continue
            
        # Get video ID from frames frames[0] -> video_1000_frame_...
        frames = video_in_meta.get('frames', [])
        if not frames:
            logger.warning(f"No frames for {video_in_meta.get('video_name')}")
            continue
            
        # Extract ID (e.g. video_1000)
        first_frame = frames[0]
        video_id = first_frame.split('_frame_')[0]
        
        split = video_in_meta.get('split', 'train') # default to train if missing
        label = video_in_meta.get('class', 'real')
        
        # Raw video path
        # Metadata path is relative to project root, e.g. data\raw\..
        raw_rel_path = video_in_meta.get('video_path')
        raw_video_path = project_root / raw_rel_path
        
        if not raw_video_path.exists():
            # Try to fix path if needed (sometimes slash diffs)
            # But let's assume it's correct relative to root
            # Debug check
            if not raw_video_path.exists():
                 logger.warning(f"Raw video missing: {raw_video_path}")
                 skipped_count += 1
                 continue

        # Output path
        output_dir = project_root / 'data' / 'processed' / 'audio' / split / label
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{video_id}.wav"
        output_path = output_dir / output_filename
        
        if output_path.exists():
            skipped_count += 1
            continue
            
        try:
            with VideoFileClip(str(raw_video_path)) as video:
                audio = video.audio
                if audio is None:
                    # Create silent audio if missing? 
                    # Dataset might expect audio.
                    # For now just skip.
                    logger.warning(f"No audio track in {raw_video_path.name}")
                    error_count += 1
                    continue
                    
                audio.write_audiofile(
                    str(output_path),
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    ffmpeg_params=["-ac", "1"],
                    verbose=False,
                    logger=None
                )
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed {video_id} ({raw_video_path.name}): {e}")
            error_count += 1
            
    logger.info(f"Audio Extraction Complete.")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped (Exists/Missing): {skipped_count}")
    logger.info(f"Errors: {error_count}")

if __name__ == "__main__":
    PROJECT_ROOT = r"d:\project\DeepFakeDetection\model-service"
    METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "preprocessing_metadata.json")
    
    extract_audio_from_metadata(METADATA_PATH, PROJECT_ROOT)
