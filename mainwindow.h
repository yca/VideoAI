#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindowPriv;
class QListWidgetItem;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pushEvaluate_clicked();
	void scriptTextChanged(const QString &text);

	void on_actionInit_Commands_triggered();

	void on_listHistory_itemDoubleClicked(QListWidgetItem *item);

	void on_actionScripts_Editor_triggered();

protected:
	void closeEvent(QCloseEvent *);
	void addScriptObject(QObject *obj, const QString &name);
	void evaluateScript(const QString &text);

private:
	Ui::MainWindow *ui;
	MainWindowPriv *p;
	QHash<QString, QObject *> scriptObjects;
};

#endif // MAINWINDOW_H
