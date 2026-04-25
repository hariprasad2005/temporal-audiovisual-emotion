class DatasetMapper:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()
        self.emotion_mapping = self._get_emotion_mapping()
    
    def _get_emotion_mapping(self) -> Dict[str, int]:
        base_mapping = {
            'happy': 0,
            'sad': 1, 
            'angry': 2,
            'surprise': 3,
            'neutral': 4
        }
        
        if self.dataset_name == 'crema_d':
            return {
                'happy': 0, 'happiness': 0,
                'sad': 1, 'sadness': 1,
                'angry': 2, 'anger': 2,
                'surprise': 3,
                'neutral': 4,
                'fear': 1,  # Map fear to sad
                'disgust': 2  # Map disgust to angry
            }
        elif self.dataset_name == 'ravdess':
            return {
                'happy': 0, '01': 0,
                'sad': 1, '04': 1,
                'angry': 2, '05': 2,
                'surprise': 3, '07': 3,
                'neutral': 4, '03': 4
            }
        elif self.dataset_name == 'afew':
            return {
                'happy': 0,
                'sad': 1,
                'angry': 2,
                'surprise': 3,
                'neutral': 4,
                'fear': 1,  # Map fear to sad
                'disgust': 2  # Map disgust to angry
            }
        else:
            return base_mapping
    
    def get_emotion_mapping(self) -> Dict[str, int]:
        return self.emotion_mapping
