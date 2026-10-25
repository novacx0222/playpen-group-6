# WINDOW -- CANVAS_WIDTH/HEIGHT are the *preferred* window size; the window is
# shrunk to fit the screen (minus the padding below for menu bar, title bar,
# Dock), and everything is laid out from the size the window actually ends up
CANVAS_WIDTH = 1500
CANVAS_HEIGHT = 1000
ROOM_PORTION = 2 / 3
INFO_PORTION = 1 - ROOM_PORTION
SCREEN_PADDING_X = 40
SCREEN_PADDING_Y = 140
MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 750  # the tallest result panel fits at this height
FONT_SIZE = 16
TITLE_FONT_SIZE = 20
SMALL_FONT_SIZE = 11
# the room is scaled to fit whatever space the window gives it, never larger
# than MAX_ROOM_SCALE pixels per unit, with ROOM_MARGIN_UNITS of empty floor
# around it so enclosures that stick out past the room are still visible
MAX_ROOM_SCALE = 40
ROOM_MARGIN_UNITS = 5

## COLORS
CANVAS_BG = "#F0F4F8"
ROOM_FILL = "#DCE8F5"
ROOM_OUTLINE = "#2B4C6F"
WALL_COLOR = "#3A3A3A"
GATE_COLOR = "#C0392B"
GATE_GAP_COLOR = "#F0F4F8"
VERTEX_COLOR = "#1A1A1A"
FACE_OVERLONG_COLOR = "#E74C3C"
GOOD_COLOR = "#1E8449"
BAD_COLOR = "#C0392B"
GHOST_COLOR = "#E08A82"  # a wall that has no connector to orient it
TRANSLUCENT_ALPHA = 0.3  # opacity of a piece the inventory can't supply
INFO_BG = "#FFFFFF"
TEXT_COLOR = "#1A1A2E"  # always set explicitly on canvas text -- the Tk
# default ("systemTextColor") can resolve to a color invisible against a
# hardcoded canvas background (e.g. white-on-white in dark mode)
MUTED_TEXT_COLOR = "#5A6B7A"
DIVIDER_COLOR = "#E0E6ED"
SECTION_HEADER_COLOR = "#2B4C6F"
STAT_VALUE_COLOR = "#14324D"
TOOLTIP_BG = "#FFFDE7"
INVENTORY_BOX_HEIGHT = 96  # pixels: small on purpose, it scrolls
INVENTORY_CHOSEN_BG = "#FFF1B8"  # an inventory row the enclosure uses
NEGATIVE_BG = "#FADBD8"  # behind a negative score, and a row the inventory is short of
HOVER_RADIUS = 12  # pixels: how close the mouse must be to a joint to pick it

# LOGIC / RULES (see Project 2: Playpen spec)
CPU_TIME_LIMIT_SECONDS = 300  # per run: constructing the player + build_enclosure()
BASELINE_SCORE = 1000.0  # what a correct enclosure starts from
INVALID_SCORE = -1000.0  # what an incorrect one gets
MIN_WALL_LENGTH = 5  # smallest possible wall/gate length, in units
MAX_FACE_LENGTH = (
    30  # longest a single linear face (run of 180-degree-joined walls) may be
)
MIN_ENCLOSURE_PIECES = 3  # fewest walls/gates needed to close a loop

# the three physical connector types and the internal angle(s) each can form
# (each connector can be flipped to its "reflex" orientation)
CONNECTOR_INTERNAL_ANGLES = {
    "straight": {False: 180.0},
    "right": {False: 90.0, True: 270.0},
    "diagonal": {False: 135.0, True: 225.0},
}

# CONSTANTS
TOL = 1e-4  # floating point tolerance for closure / containment checks
ANGLE_TOL = 1e-3  # degrees
