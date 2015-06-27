#ifndef SCRIPTMANAGER_H
#define SCRIPTMANAGER_H

#include <QHash>
#include <QObject>
#include <QString>

class ScriptEdit;
class QScriptEngine;
class WindowManager;
class ScriptManagerPriv;

class ScriptManager : public QObject
{
	Q_OBJECT
public:
	explicit ScriptManager(QObject *parent = 0);
	static ScriptManager * instance() { return inst;}
	void evaluateScript(const QString &text);
signals:

public slots:
	void setCurrentWindow(int id);
	void setCurrentCell(int row, int col);
protected:
	void addScriptObject(QObject *obj, const QString &name);

	static ScriptManager *inst;
	ScriptManagerPriv *p;
	QHash<QString, QObject *> scriptObjects;
	//ScriptEdit *edit;
};

#endif // SCRIPTMANAGER_H
