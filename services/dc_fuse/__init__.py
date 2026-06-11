'''
@Description : 直流熔丝检测场景（服务内场景形态，非插件）
'''

from .business_logic import DCFuseDetectorAPI
from .dc_fuse_detect import DCFuseDetector

__all__ = ["DCFuseDetectorAPI", "DCFuseDetector"]
