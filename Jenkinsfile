pipeline {
    agent {
        docker {
            image 'pytorch/pytorch'
        }
    }
    stages {
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
            }
        }
    }
    post {
        always {
            junit '*.xml'
        }
    }
}
