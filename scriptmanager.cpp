#include "scriptmanager.h"
#include "windowmanager.h"
#include "common.h"
#include "debug.h"

#include "opencv/opencv.h"

#include "widgets/imagewidget.h"

#include <QScriptEngine>

Q_DECLARE_METATYPE(Mat)

ScriptManager * ScriptManager::inst = NULL;

static void addQObject(QScriptEngine *e, QObject *obj, const QString &oname)
{
	QScriptValue iv = e->newQObject(obj);
	e->globalObject().setProperty(oname, iv);
	ffDebug() << oname << obj;
}

class ScriptManagerPriv
{
public:
	ScriptManagerPriv()
	{

	}

	QScriptEngine *eng;
};

QScriptValue matToScriptValue(QScriptEngine *eng, const Mat &m)
{
	OpenCV *cvmat = new OpenCV(m);
	QScriptValue obj = eng->newQObject(cvmat, QScriptEngine::ScriptOwnership);
	///QScriptValue obj = eng->newObject();
	obj.setProperty("rows", m.rows);
	obj.setProperty("cols", m.cols);
	return obj;
}

void scriptToMat(const QScriptValue &obj, Mat &m)
{
	if (obj.isObject()) {
		OpenCV *cvmat = (OpenCV *)obj.toQObject();
		m = cvmat->getRefMat();
	}
	//m.rows = obj.property("rows").toInt32();
	//m.cols = obj.property("cols").toInt32();
}

ScriptManager::ScriptManager(QObject *parent) :
	QObject(parent),
	p(new ScriptManagerPriv)
{
	inst = this;
}

void ScriptManager::setScriptEngine(QScriptEngine *eng)
{
	p->eng = eng;
	qScriptRegisterMetaType(p->eng, matToScriptValue, scriptToMat);
	addQObject(p->eng, new OpenCV(this), "cv");
	addQObject(p->eng, new Common(this), "cmn");
}

void ScriptManager::setWindowManager(WindowManager *m)
{
	wm = m;
}

void ScriptManager::setCurrentWindow(int id)
{
	wm->setCurrentImageWindow(id);
	addQObject(p->eng, wm->getCurrentImageWindow(), "iw");
}

void ScriptManager::setCurrentCell(int row, int col)
{
	wm->getCurrentImageWindow()->setCurrentCell(row, col);
}
