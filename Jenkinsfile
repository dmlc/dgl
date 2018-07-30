pipeline {
    agent {
        docker {
            image 'pytorch/pytorch'
        }
    }
    stages {
        stage('SETUP') {
            steps {
                sh 'easy_install nose'
                sh 'apt-get update && apt-get install -y libxml2-dev'
            }
        }
        stage('BUILD') {
            steps {
                dir('python') {
                    sh 'python setup.py install'
                }
            }
        }
        stage('TEST') {
            steps {
                sh 'nosetests tests -v --with-xunit'
                sh 'nosetests tests/pytorch -v --with-xunit'
            }
        }
    }
    post {
        always {
            junit '*.xml'
        }
    }
}
