from setuptools import setup, find_packages

setup(
    name='ai_posture_monitor',
    version='0.0.16',
    packages=find_packages(),
    description='A package designed to predict static pose and detect falls with 2D RGB Camera in well lit indoor environments.',
    long_description="""# AI Posture Monitor
This project introduces an innovative, cost-effective solution for real-time activity monitoring of elderly individuals. By leveraging the MediaPipe pose estimation model, fuzzy logic, and finite state machines, the system can reliably track individuals, recognize static postures (standing, sitting, lying), and detect transitions, particularly focusing on falls. 

## GITHUB: https://github.com/pat2echo/

## Features
- **Real-Time Activity Monitoring**: Tracks and analyzes movements continuously.
- **Pose Detection**: Identifies key postures**: standing, sitting, and lying down.
- **Fall Detection**: Detects falls with minimal false alarms.
- **Fuzzy Logic Analysis**: Utilizes fuzzy logic for accurate movement interpretation.
- **User-Centric Design**: Tailored for elderly individuals living alone.
- **Environmental Adaptability**: Functions effectively in well-lit indoor settings.
- **Scalable and Cost-Effective**: Affordable solution for diverse applications.
""",
    long_description_content_type='text/markdown',
    author='Patrick Ogbuitepu',
    author_email='pat2echo@gmail.com',
    keywords=['Fall Detection', 'Human Motion Classification', 'Pose Estimation', 'Elderly Care', 'Computer Vision', 'Machine Learning', 'Real-time Monitoring', 'Occlusion Handling', 'Finite State Machine', 'Activity Recognition', 'Elderly Care Technology', 'Pose Estimation Models', 'Real-Time Motion Tracking', 'Occlusion Handling', 'Static Pose Classification', 'Fuzzy Logic Systems', 'Temporal Pose Analysis', 'Human Pose Detection', 'Motion Detection Algorithms', 'Inactivity Monitoring', 'Rehabilitation Monitoring', 'Bounding Box Analysis', 'Multi-Camera Systems', 'Pose Transition Detection', 'Computational Intelligence Applications', 'Human-Centric AI', 'Assistive Technologies', 'Vision-Based Health Monitoring', 'Sensorless Activity Recognition', 'Human-Computer Interaction (HCI)', 'Machine Learning in Elderly Care', 'AI for Healthcare Monitoring', 'Zero False Alarm Systems', 'Dim Light Pose Recognition', 'Occlusion Robustness', 'Home Monitoring Systems'],
    install_requires=[
        'numpy',  # Add your dependencies here
        'opencv-python',
        'mediapipe',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ]
)