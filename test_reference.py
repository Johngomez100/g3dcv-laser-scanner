"""Verify the requested P1 axes independently using a projected marker."""
import cv2
import numpy as np

from scanner.geometry import p1_reference


def test_p1_corners_and_outward_depth():
    k = np.array([[900., 0., 640.], [0., 900., 480.], [0., 0., 1.]])
    model = np.array([[0., 0., 0.], [.23, 0., 0.], [.23, .13, 0.], [0., .13, 0.]])
    rvec = np.array([.2, -.15, .05])
    translation = np.array([-.1, -.2, 1.2])
    rotation, _ = cv2.Rodrigues(rvec)
    image, _ = cv2.projectPoints(model, rvec, translation, k, None)
    origin, basis = p1_reference(image.reshape(4, 2), k, .23, .13)
    camera_corners = model @ rotation.T + translation
    assert np.allclose((camera_corners - origin) @ basis, model, atol=1e-8)
    toward_object = translation - .1 * rotation[:, 2]
    assert np.allclose((toward_object - origin) @ basis, [0., 0., .1], atol=1e-8)
    assert np.isclose(np.linalg.det(basis), -1.)
