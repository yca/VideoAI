/****************************************************************************
**
** Copyright (C) 2007-2008 Trolltech ASA. All rights reserved.
**
** This file is part of the Qt Script Debug project on Trolltech Labs.
**
** This file may be used under the terms of the GNU General Public
** License version 2.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of
** this file.  Please review the following information to ensure GNU
** General Public Licensing requirements will be met:
** http://www.trolltech.com/products/qt/opensource.html
**
** If you are unsure which license is appropriate for your use, please
** review the following information:
** http://www.trolltech.com/products/qt/licensing.html or contact the
** sales department at sales@trolltech.com.
**
** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
**
****************************************************************************/

#ifndef SCRIPTRUNNER_SCRIPTEDIT_H
#define SCRIPTRUNNER_SCRIPTEDIT_H

#include "textedit.h"

class QListView;
class QCompleter;
class QStringListModel;
class ScriptHighlighter;

class ScriptEdit : public TextEdit
{
	Q_OBJECT

public:
	explicit ScriptEdit(QWidget *parent = 0);
	~ScriptEdit();
	void insertCompletion(const QString &name, const QStringList &list);
	void setHistory(const QStringList &h) { history = h; }
signals:
	void newEvaluation(const QString &line);
protected:
	void keyPressEvent(QKeyEvent *ev);
private:
	void commandEntered(const QString &line);
	void insertPrompt();
	void replaceCurrentLine(const QString &text);

	int histPos;
	QStringList history;
	ScriptHighlighter *highlighter;
	QCompleter *completer;
	int m_executingLineNumber;
	QListView *popup;
	QStringListModel *model;
	QHash<QString, QStringList > completions;
};


#endif // SCRIPTRUNNER_SCRIPTEDIT_H
