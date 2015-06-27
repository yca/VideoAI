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

private:
	Ui::MainWindow *ui;
	MainWindowPriv *p;
};

#endif // MAINWINDOW_H
