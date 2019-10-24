"""
Given two 2D polygons write a function that calculates the IoU of their areas,
defined as the area of their intersection divided by the area of their union.
The vertices of the polygons are constrained to lie on the unit circle and you
can assume that each polygon has at least 3 vertices, given and in sorted order.

- You are free to use basic math functions/libraries (sin, cos, atan2, numpy etc)
  but not geometry-specific libraries (such as shapely).
- You are free to look up geometry-related formulas, optionally copy paste in
  short code snippets and adapt them to your needs.
- We do care and evaluate your general code quality, structure and readability
  but you do not have to go crazy on docstrings.
"""
import numpy as np
from math import atan2, sqrt


def cleanPoly(poly):
    """ Clean a polygon list of vertices from duplicated points.

        Args:
            poly (list): polygon as a list of points (tuples).

        Returns:
            cleanedPoly (list): cleaned polygon as a list of points (tuples).
        """
    cleanedPoly = []
    for point in poly:
        if isNotInList(point, cleanedPoly):
            cleanedPoly.append(point)

    return cleanedPoly


def computeIntersection(p11, p12, p21, p22, tol=1e-6):
    """ Compute the intersection point of two lines.

        Args:
            p11 (tuple): first point of the first line.
            p12 (tuple): second point of the first line.
            p21 (tuple): first point of the second line.
            p22 (tuple): second point of the second line.
            tol (float): tolerance for parallelism and vertical lines.

        Returns:
            x, y (tuple): coordinates of the intersection point.
        """
    # Compute the differences in x coordinates for the two lines
    dx1, dx2 = p11[0] - p12[0], p21[0] - p22[0]

    # If both differences in x are below tolerance, the lines are vertical and parallel => no intersection
    if abs(dx1) < tol and abs(dx2) < tol:
        return None

    # If just the first line difference in x is below tolerance, the first line is vertical
    elif abs(dx1) < tol:
        x = (p11[0] + p12[0])/2         # the x coordinate of intersection
        m2, b2 = computeLine(p21, p22)  # get second line parameters
        return x, m2 * x + b2

    # If just the first line difference in x is below tolerance, the first line is vertical
    elif abs(dx2) < tol:
        x = (p21[0] + p22[0]) / 2       # the x coordinate of intersection
        m1, b1 = computeLine(p11, p12)  # get first line parameters
        return x, m1 * x + b1

    # If none of the differences in x difference is below tolerance, none of the lines is vertical
    else:

        m1, b1 = computeLine(p11, p12)  # get first line parameters
        m2, b2 = computeLine(p21, p22)  # get second line parameters
        dm = m1 - m2                    # difference in slope

        # If the difference in slope is below tolerance, the two lines are parallel => no intersection
        if abs(dm) < tol:
            return None
        # Else, compute the intersection x, y
        else:
            return (b2 - b1) / dm, (m1 * b2 - b1 * m2) / dm


def computeLine(p1, p2):
    """ Compute the line parameters m and b given two points.
        The equation of the line is simply y = m * x + b.

        Args:
            p1 (tuple): first point.
            p2 (tuple): second point.

        Returns:
            m, b (tuple): line parameters, slope and intercept respectively.
        """
    dx = p1[0] - p2[0]
    return (p1[1] - p2[1]) / dx, (p1[0] * p2[1] - p2[0] * p1[1]) / dx


def distanceBetween(p1, p2):
    """ Euclidean distance between two points.

       Args:
           p1 (tuple): first point.
           p2 (tuple): second point.

       Returns:
           distance (float): distance between the two points.
       """
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def iou(poly1, poly2):
    """Compute the IoU of two convex (assumptions) polygons.

        The IoU is defined as the area of their intersection divided by the area of their union.
        The vertices of the polygons are constrained to lie on the unit circle and we
        assume that each polygon has at least 3 vertices, given and in sorted order.

        Args:
            poly1 (list): first polygon as a list of points (tuples).
            poly2 (list): second polygon as a list of points (tuples).

        Returns:
            iou (float): the IoU of the two convex polygons.
        """

    poly1, poly2 = cleanPoly(poly1), cleanPoly(poly2)  # clean polygons of duplicated points
    poly3 = polyIntersection(poly1, poly2)             # compute the intersected polygon

    # If this intersection exists (i.e. it is a convex polygon with at least 3 vertices)
    if poly3:
        # Convert polygons to np arrays
        poly1 = np.array(poly1, dtype=np.float32)
        poly2 = np.array(poly2, dtype=np.float32)
        poly3 = np.array(poly3, dtype=np.float32)

        # Compute the area of intersection, i.e. the area of the intersected polygon
        intersectionArea = polyArea(poly3[:, 0], poly3[:, 1])

        # Compute the area of the union = area of the two polygons - area of the intersection
        unionArea = polyArea(poly1[:, 0], poly1[:, 1]) + polyArea(poly2[:, 0], poly2[:, 1]) - intersectionArea

        # IoU = area of intersection / area of the union
        return intersectionArea / unionArea

    # Else, the polygons do not intersect, so IoU = 0.0
    else:
        return 0.0


def isNotInList(point, list, tol=1e-6):
    """Check if point is not in a list of points already.

        Args:
            point (tuple): point to check.
            list (list): list to search.
            tol (float): distance tolerance between two points.

        Returns:
            notInList (bool): True if in the point is not already in the list.
        """

    for p in list:
        if distanceBetween(point, p) < tol:
            return False

    return True


def polyArea(x, y):
    """ Compute area of a polygon given the x and y coordinates of its vertices.

        Shoelace formula for computing the area of a polygon given the ordered list (counterclockwise or
        clockwise) of the vertices coordinates (https://en.wikipedia.org/wiki/Shoelace_formula).

        Args:
            x (np.array): ordered x coordinates of the polygon.
            y (np.array): ordered y coordinates of the polygon.

        Returns:
            area (float): the area of the polygon.
        """
    return 0.5 * np.abs(np.dot(y, np.roll(x, 1)) - np.dot(x, np.roll(y, 1)))


def polyIntersection(poly1, poly2):
    """ Compute the intersection polygons between two convex polygons.

        Args:
            poly1 (list): first polygon as a list of points (tuples).
            poly2 (list): second polygon as a list of points (tuples).

        Returns:
            intesectionPoly (list): intersection polygon as a list of points (tuples).
        """

    intersections, orientations = [], []  # list of intersection points and respective orientations wrt the origin
    n1, n2 = len(poly1), len(poly2)  # number of vertices of the two polygons

    # For each vertex in the first polygon
    for i, currentVertex1 in enumerate(poly1):

        previousVertex1 = poly1[(i + n1 - 1) % n1]  # previous vertex of the first polygon

        # Bounding box of the current edge of the first polygon
        xMax = max(currentVertex1[0], previousVertex1[0])
        xMin = min(currentVertex1[0], previousVertex1[0])
        yMax = max(currentVertex1[1], previousVertex1[1])
        yMin = min(currentVertex1[1], previousVertex1[1])

        # For each vertex in the second polygon
        for j, currentVertex2 in enumerate(poly2):

            previousVertex2 = poly2[(j + n2 - 1) % n2]  # previous vertex of the second polygon

            # Compute the intersection between the two lines of the two polygons
            intersect = computeIntersection(currentVertex1, previousVertex1, currentVertex2, previousVertex2)

            # If this intersection exists, it is in the bounding box and has not already been accounted
            if intersect:
                if xMin <= intersect[0] <= xMax and yMin <= intersect[1] <= yMax:
                    if isNotInList(intersect, intersections):
                        intersections.append(intersect)  # append it to the list
                        orientations.append(atan2(intersect[1], intersect[0]))  # append the corresponding orientation

    # If we have found fewer than 3 vertices
    if len(intersections) < 3:
        return None  # it is not a polygon and the intersection is null
    else:
        # Sort the vertices of the polygon by orientation
        intesectionPoly = [x for _, x in sorted(zip(orientations, intersections))]
        return intesectionPoly


# --------------------------------------------------------


if __name__ == "__main__":

    cases = []
    # Case 1: a vanilla case (see https://imgur.com/a/dSKXHPF for a diagram)
    poly1 = [
        (-0.7071067811865475, 0.7071067811865476),
        (0.30901699437494723, -0.9510565162951536),
        (0.5877852522924729, -0.8090169943749476),
    ]
    poly2 = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (0.7071067811865475, -0.7071067811865477),
    ]
    cases.append((poly1, poly2, "simple case", 0.12421351279682288))
    # Case 2: another simple case
    poly1 = [
        (1, 0),
        (0, 1),
        (-0.7071067811865476, -0.7071067811865476),
    ]
    poly2 = [
        (-0.1736481776669303, 0.984807753012208),
        (-1, 0),
        (0, -1),
    ]
    cases.append((poly1, poly2, "simple case 2", 0.1881047657147776))
    # Case 3: yet another simple case, note the duplicated point
    poly1 = [
        (0, -1),
        (-1, 0),
        (-1, 0),
        (0, 1),
    ]
    poly2 = [
        (0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
    ]
    cases.append((poly1, poly2, "simple case 3", 0.38148713966109243))

    # Case 4: shared edge
    poly1 = [
        (-1, 0),
        (-0.7071067811865476, -0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (1, 0),
    ]
    poly2 = [
        (0, 1),
        (-1, 0),
        (1, 0),
    ]
    cases.append((poly1, poly2, "shared edge", 0.0))

    # Case 5: same polygon
    poly1 = [
        (0, -1),
        (-1, 0),
        (1, 0),
    ]
    poly2 = [
        (0, -1),
        (-1, 0),
        (1, 0),
    ]
    cases.append((poly1, poly2, "same same", 1.0))

    # Case 6: polygons do not intersect
    poly1 = [
        (-0.7071067811865476, 0.7071067811865476),
        (-1, 0),
        (-0.7071067811865476, -0.7071067811865476),
    ]
    poly2 = [
        (0.7071067811865476, 0.7071067811865476),
        (1, 0),
        (0.7071067811865476, -0.7071067811865476),
    ]
    cases.append((poly1, poly2, "no intersection", 0.0))


    import time
    t0 = time.time()

    for poly1, poly2, description, expected in cases:
        computed = iou(poly1, poly2)
        print('-'*20)
        print(description)
        print("computed:", computed)
        print("expected:", expected)
        print("PASS" if abs(computed - expected) < 1e-8 else "FAIL")

    # details here don't matter too much, but this shouldn't be seconds
    dt = (time.time() - t0) * 1000
    print("done in %.4fms" % dt)

