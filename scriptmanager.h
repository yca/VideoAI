#ifndef SCRIPTMANAGER_H
#define SCRIPTMANAGER_H

#include <QObject>

class QScriptEngine;
class WindowManager;
class ScriptManagerPriv;

class ScriptManager : public QObject
{
	Q_OBJECT
public:
	explicit ScriptManager(QObject *parent = 0);

	void setScriptEngine(QScriptEngine *eng);
	void setWindowManager(WindowManager *m);
signals:

public slots:
	void setCurrentWindow(int id);
	void setCurrentCell(int row, int col);
protected:
	ScriptManagerPriv *p;
	WindowManager *wm;
};

#endif // SCRIPTMANAGER_H
