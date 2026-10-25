import math
import random
import tkinter as tk
from tkinter import font as tkfont

import src.constants as c
from players.player0 import Player, PlayerException
from players.registry import get_player_class
from src.args import Args
from src.enclosure import (
    Construction,
    explain_score,
    inventory_usage,
    score_construction,
    sketch_points,
    unavailable_items,
    validate_construction,
)
from src.limits import CpuLimit, PlayerTimeout
from src.scenario import Scenario, generate_scenario, read_scenario, write_scenario

# Canvas tags. The enclosure is wiped and redrawn on every click and on every
# resize; the result panel is wiped and redrawn on every click.
ENCLOSURE_TAG = "enclosure"
STATIC_TAG = "static"


def player_label(name: str | None) -> str:
    """What to call the selected player on screen: "Player 3" for a group's."""
    if name is None:
        return "No player (sandbox)"
    if name.isdigit():
        return f"Player {name}"
    if name == "e":
        return "Example player"
    if name == "i":
        return "Invalid-examples demo"
    if name.startswith("i") and name[1:].isdigit():
        return f"Invalid example {name[1:]}"
    return name


RESULT_TAG = "result"

TOOLTIP_TAG = "tooltip"

PANEL_MARGIN = 20  # left/right padding inside the info panel, in pixels
SCROLLBAR_SPACE = 16  # width reserved for the panel's scrollbar, in pixels


def _signed(x: float) -> str:
    """+90.0 / \u221284.0 -- always shows the sign, with a real minus."""
    return ("+" if x >= 0 else "\u2212") + f"{abs(x):.1f}"


def _plain(x: float) -> str:
    return ("\u2212" if x < 0 else "") + f"{abs(x):.1f}"


def _factor(x: float) -> str:
    return f"{x:g}".replace("-", "\u2212")


def _blend(color: str, background: str, alpha: float) -> str:
    """`color` at `alpha` opacity over `background`, as a hex color. A Tk
    canvas can't draw real transparency, so we mix the two colors ourselves."""
    fg = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    bg = [int(background[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(f * alpha + b * (1 - alpha)):02x}" for f, b in zip(fg, bg)
    )


def _joint_label(
    index: int,
    point: tuple[float, float],
    connectors,
    not_in_inventory: bool = False,
) -> str:
    """What the hover tooltip says about the joint at vertex `index`."""
    title = f"vertex {index}" + (" (start)" if index == 0 else "")
    if index < len(connectors):
        connector = connectors[index]
        joint = f"{connector.connector_type.value} connector, {connector.internal_angle():.0f}\u00b0"
        if not_in_inventory:
            joint += "\nnot in inventory"
    else:
        joint = "no connector"
    return f"{title}: ({point[0]:.2f}, {point[1]:.2f})\n{joint}"


class Game:
    def get_player_class(self) -> type[Player]:
        return get_player_class(self.args.player)

    # ---------------------------------------------------------------- setup

    def load_scenario(self) -> Scenario:
        if self.args.scenario_path:
            return read_scenario(self.args.scenario_path)
        scenario = generate_scenario(self.args.seed, difficulty=self.args.difficulty)
        if self.args.export_scenario:
            write_scenario(self.args.export_scenario, scenario)
        return scenario

    def __init__(self, args: Args):
        random.seed(args.seed)
        self.args = args
        self.scenario = self.load_scenario()
        self.result = None
        self.score = None
        self.construction = None
        # what is currently drawn over the room, so a resize can redraw it
        self._drawn: tuple[Construction, bool] | None = None
        # every joint currently drawn as (x_px, y_px, tooltip text), and which
        # one the mouse is over
        self._vertex_hits: list[tuple[float, float, str]] = []
        self._hovered: int | None = None
        # The player is built on the first run, not here, so that its
        # constructor counts toward the CPU time limit like the rest of it.
        self.player: Player | None = None
        self.cpu_used: float | None = None  # CPU time of the last run
        self.timeout_message: str | None = None  # set if the last run timed out
        self.inventory_box: tk.Text | None = None  # the scrollable inventory list

        if self.args.gui:
            self._build_gui()
            self._show_window()
            self.root.mainloop()
        elif not self.args.sandbox:
            self.play()

    # ------------------------------------------------------------------ gui

    def _info_width(self) -> int:
        return int(c.CANVAS_WIDTH * c.INFO_PORTION) + SCROLLBAR_SPACE

    def _panel_width(self) -> float:
        """Width available to text inside the info panel: its width, minus the
        10px padding on each side of the canvas and our own margins."""
        return self._info_width() - SCROLLBAR_SPACE - 20 - 2 * PANEL_MARGIN

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.withdraw()  # stay hidden until fully built and drawn
        self.root.title(f"Playpen - {player_label(self.args.player)}")

        # The info panel is packed FIRST. Tk's pack gives space to widgets in
        # the order they were packed, so if the window ever ends up smaller
        # than we asked for, it's the room view that shrinks -- the panel is
        # never the thing that gets cut off.
        self.right_frame = tk.Frame(self.root, bg=c.INFO_BG, width=self._info_width())
        self.right_frame.pack(side="right", fill="y")
        self.right_frame.pack_propagate(False)

        self.panel_scrollbar = tk.Scrollbar(self.right_frame, orient="vertical")
        self.panel_scrollbar.pack(side="right", fill="y", pady=10)
        self.info = tk.Canvas(
            self.right_frame,
            bg=c.INFO_BG,
            highlightthickness=0,
            yscrollincrement=20,
            yscrollcommand=self.panel_scrollbar.set,
        )
        self.panel_scrollbar.configure(command=self.info.yview)
        self.info.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        # If the window is too short for everything in the panel (a small
        # screen), the panel scrolls with the mouse wheel rather than cutting
        # content off.
        self.info.bind("<Configure>", lambda _event: self._update_panel_scroll())
        # Bound on the whole app (not just the canvas) so the wheel works with
        # the pointer anywhere over the panel -- on the button, say. The
        # handler ignores the room view and the inventory box, which scrolls
        # itself.
        self.root.bind_all("<MouseWheel>", self._on_panel_wheel)
        self.root.bind_all("<Button-4>", self._on_panel_wheel)
        self.root.bind_all("<Button-5>", self._on_panel_wheel)

        self.canvas = tk.Canvas(
            self.root,
            width=int(c.CANVAS_WIDTH * c.ROOM_PORTION),
            height=c.CANVAS_HEIGHT,
            bg=c.CANVAS_BG,
            highlightthickness=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw_scene())
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", lambda _event: self._hide_tooltip())

        self._draw_static_info_panel()
        self._redraw_scene()

    def _show_window(self):
        # macOS/Tk sometimes leaves a freshly-created window blank/white, or
        # placed off-screen/behind other windows, until something forces a
        # repaint and an explicit position. So: build hidden, give it an
        # explicit on-screen geometry, and only reveal it once it's drawn.
        #
        # The size is clamped to the screen. If we asked for more than fits,
        # the OS would shrink the window itself, and anything laid out for the
        # bigger size would end up cut off.
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(c.CANVAS_WIDTH, screen_w - c.SCREEN_PADDING_X)
        win_h = min(c.CANVAS_HEIGHT, screen_h - c.SCREEN_PADDING_Y)
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2 - 30)
        self.root.minsize(
            min(c.MIN_WINDOW_WIDTH, win_w), min(c.MIN_WINDOW_HEIGHT, win_h)
        )
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def handle_play(self):
        self.root.after(50, self.play)

    # ------------------------------------------------------- room view (left)

    def _canvas_size(self) -> tuple[int, int]:
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50:  # not laid out yet: use the size it asked for
            w, h = (
                int(float(self.canvas.cget("width"))),
                int(float(self.canvas.cget("height"))),
            )
        return w, h

    def _metrics(self) -> tuple[float, float, float]:
        """(scale, x_off, y_off) that fit the room, plus a margin, into the
        canvas as it is *right now* and center it: a point (x, y) is drawn at
        (x * scale + x_off, y * scale + y_off)."""
        w, h = self._canvas_size()
        minx, miny, maxx, maxy = self.scenario.room.polygon.bounds
        span_w = (maxx - minx) + 2 * c.ROOM_MARGIN_UNITS
        span_h = (maxy - miny) + 2 * c.ROOM_MARGIN_UNITS
        scale = min(w / span_w, h / span_h, c.MAX_ROOM_SCALE)
        x_off = w / 2 - (minx + maxx) / 2 * scale
        y_off = h / 2 - (miny + maxy) / 2 * scale
        return scale, x_off, y_off

    def _redraw_scene(self):
        self.canvas.delete("all")
        self._hovered = None
        scale, x_off, y_off = self._metrics()
        self.scenario.room.draw(self.canvas, scale, x_off, y_off)
        if self._drawn is not None:
            self.draw_construction(*self._drawn)

    def draw_construction(self, construction: Construction, valid: bool):
        # Draw the walk exactly as the pieces lay out, NOT wrapped back to the
        # start -- so a loop that doesn't close visibly doesn't. This is a
        # lenient walk: even a structurally broken construction (say, too few
        # connectors) shows the attempt, with the problem marked on it.
        points, missing = sketch_points(construction)
        scale, x_off, y_off = self._metrics()
        tag = ENCLOSURE_TAG
        self._vertex_hits = []

        def to_px(pt):
            return pt[0] * scale + x_off, pt[1] * scale + y_off

        outline = c.GOOD_COLOR if valid else c.BAD_COLOR

        # Anything the inventory can't supply is NOT placed: it's drawn as a
        # translucent ghost, and labeled, instead of as a real wall/joint.
        short_pieces, short_connectors = unavailable_items(
            construction, self.scenario.inventory
        )
        translucent = _blend(c.BAD_COLOR, c.ROOM_FILL, c.TRANSLUCENT_ALPHA)
        faint_edge = _blend(c.BAD_COLOR, c.ROOM_FILL, 0.8)
        vertices_px = [to_px(pt) for pt in points[:-1]]
        center_x = sum(x for x, _ in vertices_px) / max(len(vertices_px), 1)
        center_y = sum(y for _, y in vertices_px) / max(len(vertices_px), 1)

        for i, piece in enumerate(construction.pieces):
            a = to_px(points[i])
            b = to_px(points[i + 1])
            if i in short_pieces:
                self.canvas.create_line(*a, *b, width=9, fill=translucent, tags=tag)
                self.canvas.create_line(
                    *a, *b, width=1, fill=faint_edge, dash=(4, 3), tags=tag
                )
                # label it on the outside of the loop, clear of the wall itself
                mid_x, mid_y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                out_x, out_y = mid_x - center_x, mid_y - center_y
                norm = math.hypot(out_x, out_y) or 1.0
                out_x, out_y = out_x / norm, out_y / norm
                if abs(out_x) > abs(out_y):
                    anchor = "w" if out_x > 0 else "e"
                else:
                    anchor = "n" if out_y > 0 else "s"
                kind = "gate" if piece.is_gate else "wall"
                self.canvas.create_text(
                    mid_x + out_x * 14,
                    mid_y + out_y * 14,
                    text=f"{kind} {piece.length}: none left in inventory",
                    anchor=anchor,
                    font=("Arial", c.SMALL_FONT_SIZE, "bold"),
                    fill=c.BAD_COLOR,
                    tags=tag,
                )
            elif i in missing:
                # no connector at this wall's first vertex, so nothing really
                # fixes its direction: draw it as a dashed ghost
                self.canvas.create_line(
                    *a, *b, width=3, fill=c.GHOST_COLOR, dash=(7, 5), tags=tag
                )
            elif piece.is_gate:
                # draw the wall as a dashed line with a visible "gap" in the middle
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                gap_a = (a[0] * 0.65 + mx * 0.35, a[1] * 0.65 + my * 0.35)
                gap_b = (b[0] * 0.65 + mx * 0.35, b[1] * 0.65 + my * 0.35)
                self.canvas.create_line(
                    *a, *gap_a, width=4, fill=c.GATE_COLOR, tags=tag
                )
                self.canvas.create_line(
                    *gap_b, *b, width=4, fill=c.GATE_COLOR, tags=tag
                )
                self.canvas.create_line(
                    *gap_a, *gap_b, width=2, fill=c.GATE_COLOR, dash=(4, 3), tags=tag
                )
            else:
                self.canvas.create_line(*a, *b, width=4, fill=outline, tags=tag)

            if self.args.debug:
                self.canvas.create_text(
                    (a[0] + b[0]) / 2,
                    (a[1] + b[1]) / 2 - 10,
                    text=f"{piece.length}",
                    font=("Arial", c.SMALL_FONT_SIZE),
                    fill="black",
                    tags=tag,
                )

        # a connector the inventory can't supply: a translucent ring, not a joint
        for v in short_connectors:
            if v >= len(vertices_px):
                continue  # an extra connector with no vertex to sit at
            jx, jy = vertices_px[v]
            self.canvas.create_oval(
                jx - 10,
                jy - 10,
                jx + 10,
                jy + 10,
                outline=translucent,
                width=5,
                tags=tag,
            )
            self.canvas.create_text(
                jx + 14,
                jy - 12,
                text=f"no {construction.connectors[v].connector_type.value} connector left",
                anchor="sw",
                font=("Arial", c.SMALL_FONT_SIZE, "bold"),
                fill=c.BAD_COLOR,
                tags=tag,
            )

        # flag every vertex that has no connector
        for v in missing:
            mx_px, my_px = to_px(points[v])
            self.canvas.create_oval(
                mx_px - 9,
                my_px - 9,
                mx_px + 9,
                my_px + 9,
                outline=c.BAD_COLOR,
                width=3,
                tags=tag,
            )
            self.canvas.create_text(
                mx_px + 14,
                my_px,
                text="no connector",
                anchor="w",
                font=("Arial", c.SMALL_FONT_SIZE, "bold"),
                fill=c.BAD_COLOR,
                tags=tag,
            )

        for i, pt in enumerate(points[:-1]):
            px, py = to_px(pt)
            self.canvas.create_oval(
                px - 3,
                py - 3,
                px + 3,
                py + 3,
                fill=c.VERTEX_COLOR,
                outline="",
                tags=tag,
            )
            self._vertex_hits.append(
                (
                    px,
                    py,
                    _joint_label(i, pt, construction.connectors, i in short_connectors),
                )
            )

        start_px, start_py = to_px(points[0])
        self.canvas.create_oval(
            start_px - 6,
            start_py - 6,
            start_px + 6,
            start_py + 6,
            outline="blue",
            width=2,
            tags=tag,
        )

        # if the last wall doesn't end where the first began, mark the gap
        gap = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
        if gap > c.TOL:
            end_px, end_py = to_px(points[-1])
            self.canvas.create_line(
                end_px,
                end_py,
                start_px,
                start_py,
                fill=c.BAD_COLOR,
                width=2,
                dash=(3, 3),
                tags=tag,
            )
            self.canvas.create_oval(
                end_px - 6,
                end_py - 6,
                end_px + 6,
                end_py + 6,
                fill=c.BAD_COLOR,
                outline="",
                tags=tag,
            )
            self.canvas.create_text(
                (end_px + start_px) / 2 + 12,
                (end_py + start_py) / 2,
                text=f"gap: {gap:.1f} units",
                anchor="w",
                font=("Arial", c.SMALL_FONT_SIZE, "bold"),
                fill=c.BAD_COLOR,
                tags=tag,
            )
            end = points[-1]
            self._vertex_hits.append(
                (
                    end_px,
                    end_py,
                    (
                        f"end of the last wall: ({end[0]:.2f}, {end[1]:.2f})\n"
                        f"{gap:.2f} units from the start"
                    ),
                )
            )

        _, view_h = self._canvas_size()
        self.canvas.create_text(
            12,
            view_h - 10,
            text="hover a joint to see its coordinates",
            anchor="sw",
            font=("Arial", c.SMALL_FONT_SIZE),
            fill=c.MUTED_TEXT_COLOR,
            tags=tag,
        )

    # ------------------------------------------------------------ hover tooltip

    def _hide_tooltip(self):
        self.canvas.delete(TOOLTIP_TAG)
        self._hovered = None

    def _on_hover(self, event):
        nearest, nearest_dist = None, c.HOVER_RADIUS
        for index, (px, py, _text) in enumerate(self._vertex_hits):
            dist = math.hypot(event.x - px, event.y - py)
            if dist <= nearest_dist:
                nearest, nearest_dist = index, dist
        if nearest == self._hovered:
            return  # already showing this joint's tooltip (or none, still)
        self._hide_tooltip()
        if nearest is None:
            return
        self._hovered = nearest

        px, py, text = self._vertex_hits[nearest]
        self.canvas.create_oval(
            px - 9,
            py - 9,
            px + 9,
            py + 9,
            outline=c.SECTION_HEADER_COLOR,
            width=2,
            tags=TOOLTIP_TAG,
        )
        label = self.canvas.create_text(
            0,
            0,
            text=text,
            anchor="nw",
            font=("Arial", c.SMALL_FONT_SIZE, "bold"),
            fill=c.TEXT_COLOR,
            tags=TOOLTIP_TAG,
        )
        x0, y0, x1, y1 = self.canvas.bbox(label)
        pad = 6
        box_w, box_h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        # below-right of the joint, flipped to the other side if that would
        # run out of the view
        view_w, view_h = self._canvas_size()
        left = px + 14 if px + 14 + box_w <= view_w else px - 14 - box_w
        top = py + 14 if py + 14 + box_h <= view_h else py - 14 - box_h
        # flipping isn't enough in a very narrow view: keep the box on screen
        left = max(2, min(left, view_w - box_w - 2))
        top = max(2, min(top, view_h - box_h - 2))
        box = self.canvas.create_rectangle(
            left,
            top,
            left + box_w,
            top + box_h,
            fill=c.TOOLTIP_BG,
            outline=c.SECTION_HEADER_COLOR,
            tags=TOOLTIP_TAG,
        )
        self.canvas.tag_lower(box, label)  # background goes behind the text
        self.canvas.moveto(label, left + pad, top + pad)

    # -------------------------------------------------------- info panel (right)
    #
    # The panel is laid out top to bottom by *measuring* each piece of text and
    # starting the next one below it, never by assuming how tall it is: wrapped
    # text (a long reason, a long note) takes as many lines as it needs and
    # nothing lands on top of anything else.
    #
    # Every text item also sets an explicit `fill`. Tk's default text color
    # ("systemTextColor") is theme-aware for native widgets but can resolve to
    # something invisible against a hardcoded Canvas background (e.g.
    # white-on-white in dark mode), so we never rely on it.

    def _put(
        self,
        y: float,
        text: str,
        font,
        fill: str,
        gap: float = 0,
        extra_tag: str | None = None,
    ) -> float:
        """Draw wrapped text with its top edge at `y`; return the y just below
        it, plus `gap`."""
        item = self.info.create_text(
            PANEL_MARGIN,
            y,
            text=text,
            anchor="nw",
            font=font,
            fill=fill,
            width=self._panel_width(),
            tags=(self._tag, extra_tag) if extra_tag else self._tag,
        )
        return self.info.bbox(item)[3] + gap

    def _divider(self, y: float) -> float:
        line_y = y + 5
        self.info.create_line(
            PANEL_MARGIN,
            line_y,
            PANEL_MARGIN + self._panel_width(),
            line_y,
            fill=c.DIVIDER_COLOR,
            tags=self._tag,
        )
        return line_y + 9

    def _section_header(self, y: float, text: str) -> float:
        font = ("Arial", c.SMALL_FONT_SIZE, "bold")
        return self._put(y, text.upper(), font, c.SECTION_HEADER_COLOR, gap=4)

    def _stat(self, y: float, text: str) -> float:
        font = ("Arial", c.FONT_SIZE, "bold")
        return self._put(y, text, font, c.STAT_VALUE_COLOR, gap=2)

    def _body(self, y: float, text: str) -> float:
        font = ("Arial", c.SMALL_FONT_SIZE)
        return self._put(y, text, font, c.MUTED_TEXT_COLOR, gap=4)

    def _row(
        self, y: float, label: str, value: str, fill: str, font, gap: float = 3
    ) -> float:
        """A label on the left and a right-aligned value, on one line (the
        label wraps if it must, leaving room for the value)."""
        right = PANEL_MARGIN + self._panel_width()
        value_item = self.info.create_text(
            right,
            y,
            text=value,
            anchor="ne",
            font=font,
            fill=fill,
            tags=self._tag,
        )
        vx0, _, vx1, vy1 = self.info.bbox(value_item)
        label_item = self.info.create_text(
            PANEL_MARGIN,
            y,
            text=label,
            anchor="nw",
            font=font,
            fill=fill,
            width=self._panel_width() - (vx1 - vx0) - 12,
            tags=self._tag,
        )
        return max(vy1, self.info.bbox(label_item)[3]) + gap

    def _draw_score_calculation(self, y: float, color: str) -> float:
        """Show how the score is made, term by term. The terms come from
        `explain_score` -- the same function that produces the real score --
        so this can't drift from what the simulator actually computes."""
        weights = self.scenario.weights
        result = self.result
        breakdown = explain_score(result, weights)
        small = ("Arial", c.SMALL_FONT_SIZE)
        muted = c.MUTED_TEXT_COLOR

        y = self._section_header(y, "Score calculation")
        if not breakdown.valid:
            y = self._row(
                y, "invalid enclosure", _signed(breakdown.total), c.BAD_COLOR, small
            )
            y = self._body(
                y,
                "an invalid enclosure scores \u22121000 whatever its area, perimeter or gates",
            )
        else:
            second = "yes" if result.second_gate_bonus else "no"
            y = self._row(
                y,
                "baseline (valid enclosure)",
                _signed(breakdown.baseline),
                muted,
                small,
            )
            y = self._row(
                y,
                f"area: {_factor(weights.A)} \u00d7 {result.area:.1f}",
                _signed(breakdown.area_term),
                muted,
                small,
            )
            y = self._row(
                y,
                f"perimeter: {_factor(weights.C)} \u00d7 {result.perimeter:.1f}",
                _signed(breakdown.perimeter_term),
                muted,
                small,
            )
            y = self._row(
                y,
                f"2nd gate on another face (G = {_factor(weights.G)}): {second}",
                _signed(breakdown.gate_bonus),
                muted,
                small,
            )
        y = self._divider(y)
        bold = ("Arial", c.FONT_SIZE, "bold")
        if breakdown.total >= 0:
            return self._row(y, "SCORE", _plain(breakdown.total), color, bold)

        # negative: red text on a red-tinted band, so it can't be missed
        before = set(self.info.find_withtag(RESULT_TAG))
        bottom = self._row(y, "SCORE", _plain(breakdown.total), c.BAD_COLOR, bold)
        items = set(self.info.find_withtag(RESULT_TAG)) - before
        _, y0, _, y1 = self.info.bbox(*items)
        band = self.info.create_rectangle(
            PANEL_MARGIN - 8,
            y0 - 4,
            PANEL_MARGIN + self._panel_width() + 8,
            y1 + 4,
            fill=c.NEGATIVE_BG,
            outline=c.BAD_COLOR,
            tags=RESULT_TAG,
        )
        self.info.tag_lower(band, min(items))
        return bottom + 8

    def _update_panel_scroll(self):
        """Let the info panel scroll exactly as far as its content needs (and
        not at all if everything fits)."""
        items = self.info.find_all()
        if not items:
            return
        bottom = max(self.info.bbox(item)[3] for item in items) + 12
        height = max(self.info.winfo_height(), 1)
        self.info.configure(
            scrollregion=(
                0,
                0,
                self._panel_width() + 2 * PANEL_MARGIN,
                max(bottom, height),
            )
        )
        if bottom <= height:
            self.info.yview_moveto(0)

    def _on_panel_wheel(self, event):
        widget = str(getattr(event, "widget", self.info))
        if not widget.startswith(str(self.right_frame)):
            return  # over the room view
        if self.inventory_box is not None and widget.startswith(
            str(self.inventory_box.master)
        ):
            return  # the inventory box scrolls itself
        if self.info.yview() == (0.0, 1.0):
            return  # everything fits: nothing to scroll
        num = getattr(event, "num", None)
        if num == 4:  # X11: the wheel is buttons 4 and 5
            units = -1
        elif num == 5:
            units = 1
        else:
            # macOS reports small deltas, Windows multiples of 120
            units = -event.delta // 120 if abs(event.delta) >= 120 else -event.delta
        self.info.yview_scroll(int(units), "units")

    def _build_inventory_box(self, y: float) -> float:
        """A small scrollable box listing everything in the inventory; returns
        the y below it. `_render_inventory` fills it in."""
        frame = tk.Frame(
            self.info,
            bg=c.INFO_BG,
            highlightbackground=c.DIVIDER_COLOR,
            highlightcolor=c.DIVIDER_COLOR,
            highlightthickness=1,
        )
        scrollbar = tk.Scrollbar(frame, orient="vertical")
        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=c.SMALL_FONT_SIZE)
        mono_bold = mono.copy()
        mono_bold.configure(weight="bold")
        box = tk.Text(
            frame,
            width=1,
            height=1,
            wrap="none",
            font=mono,
            bg=c.INFO_BG,
            fg=c.TEXT_COLOR,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=6,
            pady=4,
            cursor="arrow",
            takefocus=0,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.configure(command=box.yview)
        scrollbar.pack(side="right", fill="y")
        box.pack(side="left", fill="both", expand=True)

        # explicit colors on every tag: never rely on theme defaults here
        box.tag_configure(
            "heading", font=mono_bold, foreground=c.SECTION_HEADER_COLOR, spacing1=2
        )
        box.tag_configure(
            "chosen",
            font=mono_bold,
            background=c.INVENTORY_CHOSEN_BG,
            foreground=c.TEXT_COLOR,
        )
        box.tag_configure(
            "short", font=mono_bold, background=c.NEGATIVE_BG, foreground=c.BAD_COLOR
        )

        self.info.create_window(
            PANEL_MARGIN,
            y,
            anchor="nw",
            window=frame,
            width=self._panel_width(),
            height=c.INVENTORY_BOX_HEIGHT,
            tags=STATIC_TAG,
        )
        self.inventory_box = box
        self._render_inventory(None)
        return y + c.INVENTORY_BOX_HEIGHT + 8

    def _render_inventory(self, construction: Construction | None):
        """List every inventory item; highlight the ones `construction` uses
        (and, in red, any it wants more of than the inventory has)."""
        box = self.inventory_box
        if box is None:
            return
        rows = inventory_usage(construction, self.scenario.inventory)
        box.configure(state="normal")
        box.delete("1.0", "end")
        first_used = None
        for title, kind in (
            ("walls", "wall"),
            ("gates", "gate"),
            ("connectors", "connector"),
        ):
            box.insert("end", f"{title.upper()}\n", "heading")
            group = [row for row in rows if row.kind == kind]
            if not group:
                box.insert("end", "  (none)\n")
            for row in group:
                name = f"{row.key} units" if kind != "connector" else str(row.key)
                line = f"  {name:<10} x{row.available}"
                if row.used:
                    line += f"   using {row.used}"
                if row.short:
                    line += f"  ({row.short} short)"
                tag = "short" if row.short else "chosen" if row.used else ""
                if row.used and first_used is None:
                    first_used = box.index("end-1c linestart")
                box.insert("end", line.ljust(44) + "\n", tag)
        box.configure(state="disabled")
        # scroll so the first highlighted row is in view
        box.see(first_used or "1.0")
        if first_used is None:
            box.yview_moveto(0)

    def _draw_static_info_panel(self):
        self._tag = STATIC_TAG
        y = 16
        top = y
        y = self._put(y, "Playpen", ("Arial", c.TITLE_FONT_SIZE, "bold"), c.TEXT_COLOR)
        self.info.create_text(
            PANEL_MARGIN + self._panel_width(),
            (top + y) / 2,
            text=player_label(self.args.player),
            anchor="e",
            width=self._panel_width() - 120,
            font=("Arial", c.FONT_SIZE, "bold"),
            fill=c.SECTION_HEADER_COLOR,
            tags=STATIC_TAG,
        )
        y = self._divider(y)

        room = self.scenario.room
        minx, miny, maxx, maxy = room.polygon.bounds
        y = self._section_header(y, "Room size")
        y = self._stat(y, f"{maxx - minx:.1f} × {maxy - miny:.1f} units")
        y = self._body(y, f"({room.polygon.area:.1f} units²)")
        y = self._divider(y)

        y = self._section_header(y, "Inventory (highlighted = used)")
        y = self._build_inventory_box(y)
        y = self._divider(y)

        if not self.args.sandbox:
            btn = tk.Button(
                self.info,
                text="Build enclosure",
                font=("Arial", c.FONT_SIZE),
                command=self.handle_play,
            )
            self.info.create_window(
                PANEL_MARGIN, y, anchor="nw", window=btn, tags=STATIC_TAG
            )
            self.info.update_idletasks()
            y = self._divider(y + btn.winfo_reqheight())

        self.result_start_y = y

    def draw_result(self):
        self._tag = RESULT_TAG
        self.info.delete(RESULT_TAG)
        self._render_inventory(None if self.timeout_message else self.construction)
        self._draw_result_items()
        # Draw the full per-piece list first; if that makes the panel overflow,
        # draw it again with the pieces packed into one line. (Measured, not
        # guessed. Before the window is on screen its height is unknown, so
        # then assume there's room.)
        height = self.info.winfo_height()
        bbox = self.info.bbox(RESULT_TAG)
        if height >= 100 and bbox and bbox[3] + 12 > height:
            self.info.delete(RESULT_TAG)
            self._draw_result_items(compact=True)
        self._update_panel_scroll()

    def _draw_result_items(self, compact: bool = False):
        y = self._section_header(self.result_start_y, "Result")

        note = None if self.timeout_message else getattr(self.player, "note", None)
        if note:
            font = ("Arial", c.SMALL_FONT_SIZE, "bold")
            y = self._put(y, note, font, c.SECTION_HEADER_COLOR, gap=8)
        else:
            y += 4

        if self.cpu_used is not None:
            y = self._body(
                y,
                f"player CPU time: {self.cpu_used:.2f} s "
                f"(limit {self.args.cpu_limit:g} s)",
            )

        bold = ("Arial", c.FONT_SIZE, "bold")
        if self.timeout_message:
            y = self._put(y, self.timeout_message, bold, c.BAD_COLOR, gap=6)
            self._body(
                y,
                "a run that goes over the limit is stopped and scores 0, "
                "the same as returning no enclosure",
            )
            return
        if self.construction is None:
            y = self._put(
                y, "No solution generated -- score: 0", bold, c.TEXT_COLOR, gap=6
            )
            self._body(
                y,
                "returning no enclosure scores 0: safer than an invalid one, "
                "which scores \u22121000",
            )
            return

        result = self.result
        color = c.GOOD_COLOR if result.valid else c.BAD_COLOR
        status = "VALID" if result.valid else f"INVALID: {result.reason}"
        y = self._put(y, status, bold, color, gap=4)

        if result.valid:
            y = self._divider(y)
            y = self._section_header(y, "Enclosure")
            y = self._stat(y, f"area: {result.area:.1f} units²")
            y = self._stat(y, f"circumference: {result.perimeter:.1f} units")
            if result.second_gate_bonus:
                y = self._body(y, "2nd gate bonus applies")

        pieces = self.construction.pieces
        short_pieces, _ = unavailable_items(self.construction, self.scenario.inventory)
        placed = len(pieces) - len(short_pieces)
        y = self._divider(y)
        y = self._section_header(
            y,
            f"Walls placed ({placed} of {len(pieces)})"
            if short_pieces
            else f"Walls placed ({len(pieces)})",
        )
        small = ("Arial", c.SMALL_FONT_SIZE)
        if compact:
            listing = "  \u00b7  ".join(
                f"{'gate' if p.is_gate else 'wall'} {p.length}"
                + (" (not in inventory)" if i in short_pieces else "")
                for i, p in enumerate(pieces)
            )
            y = self._put(y, listing, small, c.TEXT_COLOR, gap=4)
        else:
            for i, piece in enumerate(pieces):
                kind = "gate" if piece.is_gate else "wall"
                fill = c.GATE_COLOR if piece.is_gate else c.TEXT_COLOR
                text = f"{kind}: {piece.length} units"
                if i in short_pieces:
                    text += " \u2014 not in inventory"
                    fill = c.BAD_COLOR
                y = self._put(y, text, small, fill, gap=1)

        y = self._divider(y)
        self._draw_score_calculation(y, color)

    # --------------------------------------------------------------- play

    def play(self):
        if self.args.gui:
            self._drawn = None
            self._vertex_hits = []
            self._hide_tooltip()
            self.canvas.delete(ENCLOSURE_TAG)

        limit = self.args.cpu_limit
        self.timeout_message = None
        construction = None
        cpu = CpuLimit(limit)
        try:
            # one budget for the whole run: building the player + build_enclosure()
            with cpu:
                if self.player is None:
                    self.player = (self.get_player_class())(
                        room=self.scenario.room.copy(),
                        inventory=self.scenario.inventory.copy(),
                        weights=self.scenario.weights,
                    )
                construction = self.player.build_enclosure()
        except PlayerTimeout as e:
            construction = None
            self.timeout_message = (
                f"Time limit exceeded: stopped after {e.used:.1f}s of CPU time "
                f"(limit {limit:g}s)"
            )
        except PlayerException as e:
            print(f"Player raised PlayerException: {e}")
            raise
        except Exception as e:
            print(f"Player raised an exception: {e}")
            raise
        self.cpu_used = cpu.used
        print(f"player CPU time: {cpu.used:.2f}s (limit {limit:g}s)")

        self.construction = construction

        if self.timeout_message:
            self.result = None
            self.score = 0.0
            print(f"{self.timeout_message}. No solution. SCORE: 0")
            if self.args.gui:
                self.draw_result()
            return

        if self.player.note:
            print(f"note: {self.player.note}")

        if construction is None:
            self.result = None
            self.score = 0.0
            print("No solution generated. SCORE: 0")
            if self.args.gui:
                self.draw_result()
            return

        self.result = validate_construction(
            construction, self.scenario.room, self.scenario.inventory
        )
        self.score = score_construction(self.result, self.scenario.weights)

        if self.args.gui:
            self._drawn = (construction, self.result.valid)
            self.draw_construction(construction, self.result.valid)
            self.draw_result()

        print(f"valid: {self.result.valid} ({self.result.reason})")
        if self.result.valid:
            print(
                f"area={self.result.area:.2f} perimeter={self.result.perimeter:.2f} "
                f"gates={self.result.gate_count} second_gate_bonus={self.result.second_gate_bonus}"
            )
        print(f"SCORE: {self.score:.1f}")
