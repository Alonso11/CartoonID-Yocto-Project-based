import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstBase, GstVideo

Gst.init(None)

class YoloGElement(GstBase.BaseTransform):
    __gstmetadata__ = (
        "YOLO Passthrough Element",
        "Filter/Effect/Video",
        "Simple passthrough element with YOLO name for future extension",
        "Your Name"
    )
    
    __gsttemplates__ = (
        Gst.PadTemplate.new("src",
                           Gst.PadDirection.SRC,
                           Gst.PadPresence.ALWAYS,
                           Gst.Caps.new_any()),
        Gst.PadTemplate.new("sink",
                           Gst.PadDirection.SINK,
                           Gst.PadPresence.ALWAYS,
                           Gst.Caps.new_any())
    )

    def __init__(self):
        super(YoloGElement, self).__init__()
        # You can add YOLO-specific properties here in the future
        self.set_in_place(True)  # Enable in-place processing for efficiency
    
    def do_transform_ip(self, buf):
        """
        In-place transformation method - perfect for passthrough
        This is where you would add YOLO processing logic later
        """
        # Currently just passes through the buffer unchanged
        # For future YOLO implementation, you would process the buffer here
        return Gst.FlowReturn.OK
    
    def do_set_caps(self, incaps, outcaps):
        """
        Called when caps are negotiated
        """
        # You can add YOLO-specific caps handling here if needed
        return True

# Register the element
GObject.type_register(YoloGElement)
__gstelementfactory__ = ("yolo_gelement", Gst.Rank.NONE, YoloGElement)