"""Dispose test-owned native widgets on the GUI thread after worker shutdown."""


def dispose_widget(widget):
    from PySide6.QtCore import QCoreApplication, QEvent
    destroyed = []
    widget.destroyed.connect(lambda *_: destroyed.append(True))
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    if not destroyed:
        raise AssertionError('Test-owned native widget was not destroyed on the GUI thread')
